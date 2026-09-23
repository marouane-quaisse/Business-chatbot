"""
=============================================================================
AUTHENTIFICATION — inscription, connexion, jetons
=============================================================================

CE QUE FAIT CE MODULE
---------------------
  1. Hacher les mots de passe (bcrypt) — jamais de mot de passe en clair
  2. Créer / vérifier des jetons de connexion
  3. Fournir une dépendance FastAPI qui identifie l'utilisateur courant

DEUX MODES D'UTILISATION
------------------------
  - AUTHENTIFIÉ : un token valide est fourni → l'utilisateur est identifié,
                  ses conversations sont sauvegardées en base.
  - ANONYME     : aucun token → l'utilisateur est `None`, rien n'est
                  sauvegardé. C'est le mode "Continuer sans s'authentifier".

POURQUOI PAS DE JWT ?
---------------------
Un JWT est signé mais ne peut pas être révoqué avant son expiration.
Ici, le token est une chaîne aléatoire stockée en base : on peut donc le
supprimer immédiatement (déconnexion, bannissement). C'est plus simple
ET plus sûr pour ce cas d'usage.

SÉCURITÉ
--------
  - Mot de passe : hash bcrypt (coût 12), jamais stocké en clair
  - Token : 32 octets aléatoires cryptographiquement sûrs (secrets)
  - Comparaison : on interroge la base par le token (pas de comparaison
    manuelle de chaînes, donc pas de timing attack exploitable)
=============================================================================
"""

import secrets
from datetime import timedelta

import bcrypt
from fastapi import Depends, Header, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from config import TOKEN_EXPIRE_DAYS
from database import get_db
from models import AuthToken, User, utcnow

# =============================================================================
# 1. HACHAGE DES MOTS DE PASSE
# =============================================================================
# bcrypt est volontairement LENT (contrairement à SHA-256) : cela rend les
# attaques par force brute beaucoup trop coûteuses pour être rentables.
#
# NOTE : on utilise directement la bibliothèque `bcrypt` plutôt que `passlib`.
# passlib 1.7.4 (dernière version, 2020) est incompatible avec bcrypt >= 4.1 :
# il lit `bcrypt.__about__.__version__`, attribut supprimé depuis. L'appel
# direct à bcrypt est plus simple et évite cette dépendance fragile.
_BCRYPT_ROUNDS = 12

# bcrypt refuse les entrées de plus de 72 OCTETS (et non caractères).
# Un mot de passe accentué peut donc dépasser la limite même s'il fait
# moins de 72 caractères. On tronque sur les octets pour éviter une
# exception lors de l'inscription.
_BCRYPT_MAX_BYTES = 72


def _to_bcrypt_bytes(password: str) -> bytes:
    """Encode un mot de passe en UTF-8, tronqué à 72 octets (limite bcrypt)."""
    encoded = password.encode("utf-8")
    if len(encoded) <= _BCRYPT_MAX_BYTES:
        return encoded
    # On coupe sur les octets puis on retire les octets de continuation
    # invalides (0x80-0xBF) pour ne pas produire un caractère cassé.
    truncated = encoded[:_BCRYPT_MAX_BYTES]
    while truncated and (truncated[-1] & 0xC0) == 0x80:
        truncated = truncated[:-1]
    return truncated


def hash_password(password: str) -> str:
    """Transforme un mot de passe en hash bcrypt (irréversible)."""
    hashed = bcrypt.hashpw(_to_bcrypt_bytes(password), bcrypt.gensalt(_BCRYPT_ROUNDS))
    return hashed.decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    """Vérifie qu'un mot de passe correspond bien au hash stocké."""
    try:
        return bcrypt.checkpw(_to_bcrypt_bytes(plain), hashed.encode("utf-8"))
    except (ValueError, TypeError):
        # Hash corrompu ou format inconnu : on refuse l'authentification
        # plutôt que de laisser remonter une erreur 500.
        return False


# =============================================================================
# 2. JETONS
# =============================================================================
def create_token(db: Session, user: User) -> str:
    """
    Crée un nouveau jeton pour un utilisateur et le stocke en base.

    Returns:
        Le token (à renvoyer au client, qui le stockera dans localStorage).
    """
    token = secrets.token_urlsafe(32)
    db.add(
        AuthToken(
            token=token,
            user_id=user.id,
            expires_at=utcnow() + timedelta(days=TOKEN_EXPIRE_DAYS),
        )
    )
    db.commit()
    return token


def revoke_token(db: Session, token: str) -> None:
    """Supprime un jeton (déconnexion)."""
    db_token = db.get(AuthToken, token)
    if db_token:
        db.delete(db_token)
        db.commit()


# =============================================================================
# 3. DÉPENDANCES FASTAPI
# =============================================================================
def get_current_user_optional(
    authorization: str | None = Header(default=None),
    db: Session = Depends(get_db),
) -> User | None:
    """
    Identifie l'utilisateur courant, ou renvoie None s'il est anonyme.

    C'est la dépendance utilisée par les endpoints de chat : elle accepte
    AUSSI BIEN un utilisateur connecté qu'un visiteur anonyme.

    Le client envoie le token dans l'en-tête HTTP :
        Authorization: Bearer <token>

    Returns:
        L'utilisateur si le token est valide, None sinon.
        ⚠️ Un token invalide ou expiré ne lève PAS d'erreur ici : on
        retombe simplement en mode anonyme (dégradation gracieuse).
    """
    if not authorization:
        return None

    # On accepte "Bearer xxx" (standard) et "xxx" (tolérance).
    token = authorization.removeprefix("Bearer ").strip()
    if not token:
        return None

    db_token = db.get(AuthToken, token)
    if db_token is None:
        return None

    # Vérification de l'expiration.
    # ⚠️ SQLite ne stocke pas le fuseau horaire : la date relue est "naïve".
    # On la rend comparable en la marquant explicitement en UTC.
    expires_at = db_token.expires_at
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=utcnow().tzinfo)

    if expires_at < utcnow():
        # Token expiré : on le nettoie au passage.
        db.delete(db_token)
        db.commit()
        return None

    return db.get(User, db_token.user_id)


def get_current_user(
    user: User | None = Depends(get_current_user_optional),
) -> User:
    """
    Identifie l'utilisateur courant et EXIGE qu'il soit authentifié.

    Utilisée par les endpoints qui manipulent des données personnelles
    (liste des conversations, suppression...). Renvoie 401 sinon.
    """
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentification requise.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


# =============================================================================
# 4. OPÉRATIONS MÉTIER
# =============================================================================
def get_user_by_email(db: Session, email: str) -> User | None:
    """Retrouve un utilisateur par son email (insensible à la casse)."""
    return db.scalar(select(User).where(User.email == email.strip().lower()))


def create_user(db: Session, email: str, password: str) -> User:
    """
    Crée un nouvel utilisateur.

    Raises:
        ValueError: si l'email est déjà utilisé.
    """
    email = email.strip().lower()

    if get_user_by_email(db, email):
        raise ValueError("Un compte existe déjà avec cet email.")

    user = User(email=email, hashed_password=hash_password(password))
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def authenticate(db: Session, email: str, password: str) -> User | None:
    """
    Vérifie les identifiants et renvoie l'utilisateur si corrects.

    ⚠️ On renvoie le MÊME message d'erreur que l'email soit inconnu ou le
    mot de passe faux : cela évite de révéler quels emails sont enregistrés
    (énumération de comptes).
    """
    user = get_user_by_email(db, email)
    if user is None:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user
