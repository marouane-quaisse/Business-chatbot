/* =========================================================================
   ATLAS — logique de l'interface de chat
   =========================================================================
   Quatre responsabilités :
     1. Envoyer les questions et lire le flux SSE token par token.
     2. Gérer l'authentification et la liste des conversations.
     3. Gérer la liste des documents (ajout, suppression, réindexation).
     4. Restaurer la session au chargement de la page.

   Aucune dépendance externe : tout est en JavaScript natif.
   ========================================================================= */

const $ = (id) => document.getElementById(id);

const els = {
  chat:        $("chat"),
  welcome:     $("welcome"),
  input:       $("input"),
  send:        $("send"),
  stop:        $("stop"),
  topbarMeta:  $("topbarMeta"),
  toast:       $("toast"),
  fileInput:   $("fileInput"),
  uploadBtn:   $("uploadBtn"),
  docList:     $("docList"),
  clearBtn:    $("clearBtn"),
  // --- Authentification ---
  authScreen:  $("authScreen"),
  authForm:    $("authForm"),
  authEmail:   $("authEmail"),
  authPassword:$("authPassword"),
  authSubmit:  $("authSubmit"),
  authError:   $("authError"),
  tabLogin:    $("tabLogin"),
  tabSignup:   $("tabSignup"),
  anonBtn:     $("anonBtn"),
  anonBanner:  $("anonBanner"),
  anonLoginBtn:$("anonLoginBtn"),
  app:         $("app"),
  userRow:     $("userRow"),
  userEmail:   $("userEmail"),
  logoutBtn:   $("logoutBtn"),
  switchAccountBtn: $("switchAccountBtn"),
  // --- Conversations ---
  sessionsPanel: $("sessionsPanel"),
  sessionsList:  $("sessionsList"),
  newSessionBtn: $("newSessionBtn"),
  loadMoreBtn:   $("loadMoreBtn"),
};

let isStreaming = false;
let abortController = null;
let messageCount = 0;

/* ============================ État de session ========================== */

/**
 * État global de l'application.
 *
 * token     : le jeton d'authentification (null en mode anonyme)
 * email     : l'email de l'utilisateur connecté
 * sessionId : identifiant de la conversation courante (UUID généré ici)
 * history   : historique local, utilisé UNIQUEMENT en mode anonyme
 *             (en mode authentifié, l'historique vient de la base)
 */
const state = {
  token: localStorage.getItem("atlas_token") || null,
  email: localStorage.getItem("atlas_email") || null,
  sessionId: null,
  history: [],
  authMode: "login", // "login" | "signup"
  loadedCount: 0,    // nombre de messages chargés depuis la base
  hasMore: false,    // reste-t-il des messages plus anciens ?
};

/** Génère un identifiant de conversation (UUID v4). */
function newSessionId() {
  // crypto.randomUUID() est disponible dans tous les navigateurs modernes.
  // Le fallback couvre les contextes non sécurisés (http:// sur un autre host).
  if (crypto.randomUUID) return crypto.randomUUID();
  return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, (c) => {
    const r = (Math.random() * 16) | 0;
    return (c === "x" ? r : (r & 0x3) | 0x8).toString(16);
  });
}

/** En-têtes HTTP, avec le token si l'utilisateur est connecté. */
function authHeaders(extra = {}) {
  const headers = { ...extra };
  if (state.token) headers["Authorization"] = `Bearer ${state.token}`;
  return headers;
}

/* ============================ Utilitaires ============================== */

function toast(message, kind = "") {
  els.toast.textContent = message;
  els.toast.className = `toast show ${kind}`;
  clearTimeout(toast._timer);
  toast._timer = setTimeout(() => {
    els.toast.className = "toast";
  }, 3200);
}

function scrollToBottom() {
  els.chat.scrollTop = els.chat.scrollHeight;
}

function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

/**
 * Mini-rendu Markdown.
 *
 * POURQUOI ?
 * Le LLM rédige naturellement en Markdown (**gras**, listes, `code`).
 * Sans traitement, l'utilisateur voit les astérisques brutes — peu lisible.
 *
 * SÉCURITÉ
 * On échappe TOUT le HTML AVANT d'appliquer les règles Markdown. Ainsi,
 * une réponse contenant du HTML malveillant reste affichée comme du texte
 * inerte. C'est l'ordre inverse qui crée les failles XSS.
 *
 * On ne gère volontairement que le sous-ensemble utile ici (pas de tableaux,
 * pas de liens, pas de HTML) : moins de code, moins de bugs.
 */
function renderMarkdown(raw) {
  // 1) Échappement AVANT toute transformation.
  let html = escapeHtml(raw);

  // 2) Blocs de code ```...``` → contenu préservé tel quel.
  html = html.replace(/```[\w]*\n?([\s\S]*?)```/g, (_, code) => {
    return `<pre class="code"><code>${code.replace(/\n$/, "")}</code></pre>`;
  });

  // 3) Code inline `...`
  html = html.replace(/`([^`\n]+)`/g, '<code class="inline">$1</code>');

  // 4) Titres ### / ## / #
  html = html.replace(/^#{3,}\s+(.+)$/gm, '<h4>$1</h4>');
  html = html.replace(/^#{1,2}\s+(.+)$/gm, '<h3>$1</h3>');

  // 5) Gras (**texte**) puis italique (*texte*).
  html = html.replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>");
  html = html.replace(/(^|[^*])\*([^*\n]+)\*(?!\*)/g, "$1<em>$2</em>");

  // 6) Listes à puces (-, *, •) et listes numérotées (1.).
  const lines = html.split("\n");
  const out = [];
  let listType = null;

  const closeList = () => {
    if (listType) {
      out.push(`</${listType}>`);
      listType = null;
    }
  };

  for (const line of lines) {
    const bullet = line.match(/^\s*[-*•]\s+(.*)$/);
    const numbered = line.match(/^\s*\d+\.\s+(.*)$/);

    if (bullet) {
      if (listType !== "ul") { closeList(); out.push("<ul>"); listType = "ul"; }
      out.push(`<li>${bullet[1]}</li>`);
    } else if (numbered) {
      if (listType !== "ol") { closeList(); out.push("<ol>"); listType = "ol"; }
      out.push(`<li>${numbered[1]}</li>`);
    } else {
      closeList();
      out.push(line);
    }
  }
  closeList();

  html = out.join("\n");

  // 7) Paragraphes : les lignes vides deviennent des séparateurs.
  return html
    .split(/\n{2,}/)
    .map((block) => {
      const trimmed = block.trim();
      if (!trimmed) return "";
      // On n'enveloppe pas les blocs déjà structurés (listes, titres, code).
      if (/^<(ul|ol|h3|h4|pre)/.test(trimmed)) return trimmed;
      return `<p>${trimmed.replace(/\n/g, "<br>")}</p>`;
    })
    .join("");
}

/* ============================ Rendu des messages ======================= */

const ICON_USER = "Vous".charAt(0);
const ICON_BOT = "◆";

const ICON_FILE =
  '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">' +
  '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>' +
  '<path d="M14 2v6h6"/></svg>';

/**
 * Crée la coquille d'un message et l'ajoute au fil de discussion.
 * @param {string} role  "user" ou "bot"
 * @param {string} name  libellé affiché au-dessus de la bulle
 * @param {string} text  contenu initial (vide pendant le streaming)
 * @returns les éléments internes à manipuler ensuite.
 */
function addMessage(role, name, text = "") {
  const wrapper = document.createElement("div");
  wrapper.className = `msg ${role}`;

  const avatarText = role === "user" ? ICON_USER : ICON_BOT;

  wrapper.innerHTML = `
    <div class="avatar">${avatarText}</div>
    <div class="bubble">
      <div class="bubble-name">${name}</div>
      <div class="bubble-text"></div>
    </div>
  `;

  // Inner container : garde la largeur du fil centrée et bornée.
  let inner = document.querySelector(".chat-inner");
  if (!inner) {
    inner = document.createElement("div");
    inner.className = "chat-inner";
    els.chat.appendChild(inner);
  }
  inner.appendChild(wrapper);

  const textEl = wrapper.querySelector(".bubble-text");
  if (text) textEl.textContent = text;

  messageCount++;
  updateTopbar();
  scrollToBottom();

  return { wrapper, text: textEl };
}

function updateTopbar() {
  els.topbarMeta.textContent =
    messageCount === 0 ? "" : `${Math.ceil(messageCount / 2)} échange(s)`;
}

function showThinking(textEl) {
  textEl.innerHTML = '<div class="thinking"><i></i><i></i><i></i></div>';
}

function renderSources(textEl, sources) {
  if (!sources || !sources.length) return;

  const container = document.createElement("div");
  container.className = "sources";
  container.innerHTML =
    '<span class="sources-label">Sources</span>' +
    sources
      .map(
        (s) =>
          `<span class="source-chip">${ICON_FILE}${escapeHtml(
            String(s).replace(/\.(md|txt|pdf)$/i, "")
          )}</span>`
      )
      .join("");

  // On insère le bloc sources juste après le texte, dans la même bulle.
  textEl.parentElement.appendChild(container);
}

/* ============================ Appel de l'API =========================== */

async function sendQuestion(question) {
  if (isStreaming || !question.trim()) return;

  // Le premier message fait disparaître l'écran d'accueil.
  if (els.welcome) els.welcome.style.display = "none";

  addMessage("user", "Vous", question);

  const bot = addMessage("bot", "Atlas");
  showThinking(bot.text);

  isStreaming = true;
  els.send.hidden = true;
  els.stop.hidden = false;
  els.input.disabled = true;

  abortController = new AbortController();
  let answerText = "";
  let firstToken = true;
  const startedAt = performance.now();

  // --- Historique envoyé au serveur ---
  // En mode ANONYME, c'est le SEUL moyen pour le serveur de connaître le
  // contexte : rien n'est stocké en base. En mode authentifié, le serveur
  // ignore ce champ et lit l'historique depuis la base.
  const historyPayload = state.token ? null : state.history.slice(-6);

  try {
    const response = await fetch("/api/chat/stream", {
      method: "POST",
      headers: authHeaders({ "Content-Type": "application/json" }),
      body: JSON.stringify({
        question,
        session_id: state.sessionId,
        history: historyPayload,
      }),
      signal: abortController.signal,
    });

    if (!response.ok) {
      throw new Error(`Erreur serveur (${response.status})`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    // Lecture du flux SSE : on découpe sur les doubles retours à la ligne,
    // chaque bloc étant un événement `data: {...}`.
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const blocks = buffer.split("\n\n");
      buffer = blocks.pop() || ""; // on garde le bloc incomplet

      for (const block of blocks) {
        const line = block.trim();
        if (!line.startsWith("data:")) continue;

        let event;
        try {
          event = JSON.parse(line.slice(5).trim());
        } catch {
          continue;
        }

        if (event.type === "sources") {
          renderSources(bot.text, event.data);
        } else if (event.type === "token") {
          if (firstToken) {
            bot.text.innerHTML = "";
            firstToken = false;
          }
          answerText += event.data;
          // On rend le Markdown à chaque token : le texte se met en forme
          // au fur et à mesure. Le curseur est ajouté APRÈS le rendu pour
          // ne pas être interprété comme du Markdown.
          bot.text.innerHTML =
            renderMarkdown(answerText) + '<span class="cursor"></span>';
          scrollToBottom();
        } else if (event.type === "done") {
          const elapsed = ((performance.now() - startedAt) / 1000).toFixed(1);
          console.debug("Latence totale :", elapsed, "s", event.data);
        } else if (event.type === "error") {
          throw new Error(event.data);
        }
      }
    }

    // Fin du flux : on retire le curseur clignotant.
    bot.text.innerHTML = renderMarkdown(answerText || "(aucune réponse)");

    // --- Mise à jour de l'historique local ---
    // En mode anonyme, c'est ce tableau qui porte le contexte de la
    // conversation (il est renvoyé à chaque question). En mode authentifié,
    // il ne sert qu'à l'affichage : la source de vérité est la base.
    state.history.push(["user", question]);
    state.history.push(["assistant", answerText]);
    // On borne la taille pour ne pas envoyer un historique géant.
    if (state.history.length > 20) {
      state.history = state.history.slice(-20);
    }

    // --- Rafraîchissement de la liste des conversations ---
    // La conversation vient d'être créée (ou enrichie) côté serveur : on
    // recharge la liste pour qu'elle apparaisse SANS que l'utilisateur ait
    // à rafraîchir la page. On ne le fait qu'en mode authentifié, seul mode
    // où les conversations sont persistées.
    if (state.token) {
      await loadSessions();
      // On remet en surbrillance la conversation courante : loadSessions()
      // reconstruit la liste et perd donc la classe `.active`.
      document.querySelectorAll(".session-item").forEach((el) => {
        el.classList.toggle("active", el.dataset.id === state.sessionId);
      });
    }
  } catch (error) {
    if (error.name === "AbortError") {
      bot.text.innerHTML = renderMarkdown(answerText) + "<p><em>[réponse interrompue]</em></p>";
    } else {
      bot.wrapper.classList.add("error");
      bot.text.textContent = `Une erreur est survenue : ${error.message}`;
      toast(error.message, "err");
    }
  } finally {
    isStreaming = false;
    abortController = null;
    els.send.hidden = false;
    els.stop.hidden = true;
    els.input.disabled = false;
    els.input.focus();
    scrollToBottom();
  }
}

/* ============================ Documents =============================== */

/**
 * Charge la liste des documents présents dans ressources/.
 *
 * On interroge le serveur à chaque ajout/suppression plutôt que de
 * maintenir une copie locale : la liste reste ainsi toujours fidèle au
 * contenu réel du dossier, même si un fichier a été modifié à la main.
 */
async function loadDocuments() {
  try {
    const response = await fetch("/api/documents", { headers: authHeaders() });
    if (!response.ok) throw new Error("indisponible");

    const documents = await response.json();

    if (!documents.length) {
      els.docList.innerHTML =
        '<li class="doc-empty">Aucun document. Ajoutez un fichier .pdf, .md ou .txt.</li>';
      return;
    }

    els.docList.innerHTML = documents
      .map(
        (doc) => `
        <li class="doc-item" data-name="${escapeHtml(doc.name)}" title="${escapeHtml(
          doc.name
        )}">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
            <path d="M14 2v6h6"/>
          </svg>
          <span class="doc-name">${escapeHtml(doc.name)}</span>
          <button class="doc-download" title="Télécharger" aria-label="Télécharger ${escapeHtml(
            doc.name
          )}">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M12 3v12M7 10l5 5 5-5M5 21h14"/>
            </svg>
          </button>
          <button class="doc-delete" title="Supprimer" aria-label="Supprimer ${escapeHtml(
            doc.name
          )}">×</button>
        </li>`
      )
      .join("");

    els.docList.querySelectorAll(".doc-download").forEach((button) => {
      button.addEventListener("click", (event) => {
        event.stopPropagation();
        const name = button.closest(".doc-item").dataset.name;
        downloadDocument(name);
      });
    });

    els.docList.querySelectorAll(".doc-delete").forEach((button) => {
      button.addEventListener("click", (event) => {
        event.stopPropagation();
        const name = button.closest(".doc-item").dataset.name;
        deleteDocument(name);
      });
    });
  } catch {
    els.docList.innerHTML =
      '<li class="doc-empty">Liste indisponible (serveur injoignable).</li>';
  }
}

/* ============================ Actions ================================= */

async function uploadFile(file) {
  if (!file) return;

  const formData = new FormData();
  formData.append("file", file);

  els.uploadBtn.disabled = true;
  toast(`Indexation de ${file.name}…`);

  try {
    const response = await fetch("/api/upload", {
      method: "POST",
      headers: authHeaders(),
      body: formData,
    });
    const data = await response.json();

    if (!response.ok) throw new Error(data.detail || "échec de l'envoi");

    toast(`${data.chunks_added} passages ajoutés depuis ${data.filename}`, "ok");
    // La liste des documents doit refléter l'ajout immédiatement.
    await loadDocuments();
  } catch (error) {
    toast(`Erreur : ${error.message}`, "err");
  } finally {
    els.uploadBtn.disabled = false;
    els.fileInput.value = "";
  }
}

/**
 * Supprime un document puis laisse le serveur réindexer automatiquement.
 *
 * La réindexation est faite côté serveur dans la même requête : le client
 * n'a donc rien à orchestrer, et la base ne peut pas se retrouver dans un
 * état intermédiaire (fichier supprimé mais toujours indexé).
 */
async function deleteDocument(name) {
  if (!confirm(`Supprimer « ${name} » et réindexer la base ?`)) return;

  toast(`Suppression de ${name}…`);

  try {
    const response = await fetch(`/api/documents/${encodeURIComponent(name)}`, {
      method: "DELETE",
      headers: authHeaders(),
    });
    const data = await response.json();

    if (!response.ok) throw new Error(data.detail || "échec de la suppression");

    toast(`${name} supprimé — ${data.chunks_indexed} passages réindexés`, "ok");
    await loadDocuments();
  } catch (error) {
    toast(`Erreur : ${error.message}`, "err");
  }
}

/**
 * Télécharge un document depuis le serveur.
 *
 * On ne peut pas utiliser un simple lien `<a href>` : l'endpoint exige un
 * en-tête `Authorization`, qu'un lien ne sait pas envoyer. On récupère donc
 * le fichier via `fetch`, on le transforme en Blob, puis on déclenche le
 * téléchargement avec un lien temporaire pointant sur ce Blob.
 */
async function downloadDocument(name) {
  toast(`Téléchargement de ${name}…`);

  try {
    const response = await fetch(
      `/api/documents/${encodeURIComponent(name)}/download`,
      { headers: authHeaders() }
    );
    if (!response.ok) throw new Error("échec du téléchargement");

    const blob = await response.blob();
    const url = URL.createObjectURL(blob);

    const link = document.createElement("a");
    link.href = url;
    link.download = name;
    document.body.appendChild(link);
    link.click();
    link.remove();

    // On libère la mémoire occupée par le Blob.
    URL.revokeObjectURL(url);
  } catch (error) {
    toast(`Erreur : ${error.message}`, "err");
  }
}

function clearConversation() {
  const inner = document.querySelector(".chat-inner");
  if (inner) inner.remove();

  messageCount = 0;
  state.history = [];
  state.loadedCount = 0;
  state.hasMore = false;
  els.loadMoreBtn.hidden = true;
  updateTopbar();

  if (els.welcome) {
    els.welcome.style.display = "";
    if (!els.welcome.parentElement) els.chat.appendChild(els.welcome);
  }
  els.input.focus();
}

/* ============================ Authentification ========================= */

/** Bascule entre les onglets Connexion / Inscription. */
function setAuthMode(mode) {
  state.authMode = mode;
  els.tabLogin.classList.toggle("active", mode === "login");
  els.tabSignup.classList.toggle("active", mode === "signup");
  els.authSubmit.textContent = mode === "login" ? "Se connecter" : "Créer un compte";
  els.authPassword.autocomplete = mode === "login" ? "current-password" : "new-password";
  els.authError.textContent = "";
}

/** Affiche un message d'erreur sous le formulaire. */
function showAuthError(message) {
  els.authError.textContent = message;
}

/**
 * Soumet le formulaire de connexion ou d'inscription.
 */
async function submitAuth(event) {
  event.preventDefault();

  const email = els.authEmail.value.trim();
  const password = els.authPassword.value;
  if (!email || !password) return;

  els.authSubmit.disabled = true;
  els.authError.textContent = "";

  const endpoint = state.authMode === "login" ? "/api/auth/login" : "/api/auth/signup";

  try {
    const response = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, password }),
    });

    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "Échec de l'authentification.");

    // --- Connexion réussie ---
    state.token = data.token;
    state.email = data.email;
    localStorage.setItem("atlas_token", data.token);
    localStorage.setItem("atlas_email", data.email);

    await enterApp();
  } catch (error) {
    showAuthError(error.message);
  } finally {
    els.authSubmit.disabled = false;
  }
}

/** Déconnexion : révoque le token côté serveur et revient à l'écran d'auth. */
async function logout() {
  try {
    await fetch("/api/auth/logout", { method: "POST", headers: authHeaders() });
  } catch {
    // Peu importe si l'appel échoue : on nettoie quand même le client.
  }

  showAuthScreen();
}

/**
 * Entre dans l'application (après connexion ou en mode anonyme).
 */
async function enterApp() {
  els.authScreen.hidden = true;
  els.app.hidden = false;

  const authenticated = Boolean(state.token);

  // Bandeau d'avertissement : uniquement en mode anonyme.
  els.anonBanner.hidden = authenticated;

  // Panneau des conversations : uniquement en mode authentifié.
  els.sessionsPanel.hidden = !authenticated;
  els.userRow.hidden = !authenticated;
  if (authenticated) els.userEmail.textContent = state.email;

  // Nouvelle conversation à chaque entrée.
  startNewSession();

  // Liste des documents (disponible aussi en mode anonyme).
  loadDocuments();

  if (authenticated) {
    await loadSessions();
  }

  els.input.focus();
}

/**
 * Vérifie que le token stocké est toujours valide auprès du serveur.
 *
 * Un token peut être expiré ou révoqué (déconnexion depuis un autre onglet,
 * nettoyage de la base…). Sans cette vérification, l'utilisateur entrerait
 * dans l'application avec un token mort : toutes les requêtes échoueraient
 * silencieusement en 401 et l'interface paraîtrait vide.
 *
 * @returns {Promise<boolean>} true si le token est accepté par le serveur.
 */
async function validateToken() {
  if (!state.token) return false;
  try {
    const response = await fetch("/api/auth/me", { headers: authHeaders() });
    if (!response.ok) return false;
    const data = await response.json();
    // Le serveur fait foi : on resynchronise l'email local.
    if (data.email) {
      state.email = data.email;
      localStorage.setItem("atlas_email", data.email);
    }
    return true;
  } catch {
    // Serveur injoignable : on ne détruit pas la session, on laisse entrer.
    return true;
  }
}

/** Efface la session locale et affiche l'écran de connexion. */
function showAuthScreen() {
  state.token = null;
  state.email = null;
  localStorage.removeItem("atlas_token");
  localStorage.removeItem("atlas_email");

  els.app.hidden = true;
  els.authScreen.hidden = false;
  els.authPassword.value = "";
  els.authError.textContent = "";
  setAuthMode("login");
  els.authEmail.focus();
}

/* ============================ Conversations ============================ */

/** Démarre une nouvelle conversation (vide l'écran, nouveau session_id). */
function startNewSession() {
  state.sessionId = newSessionId();
  state.history = [];
  state.loadedCount = 0;
  state.hasMore = false;
  els.loadMoreBtn.hidden = true;

  const inner = document.querySelector(".chat-inner");
  if (inner) inner.remove();

  messageCount = 0;
  updateTopbar();

  if (els.welcome) {
    els.welcome.style.display = "";
    if (!els.welcome.parentElement) els.chat.appendChild(els.welcome);
  }

  // Désélectionne la conversation active dans la liste.
  document.querySelectorAll(".session-item").forEach((el) => el.classList.remove("active"));
}

/** Charge la liste des conversations de l'utilisateur. */
async function loadSessions() {
  try {
    const response = await fetch("/api/sessions", { headers: authHeaders() });
    if (!response.ok) return;

    const sessions = await response.json();

    els.sessionsList.innerHTML = sessions
      .map(
        (s) => `
        <li class="session-item" data-id="${s.id}">
          <span class="session-title">${escapeHtml(s.title)}</span>
          <span class="session-count">${s.message_count}</span>
          <button class="session-delete" title="Supprimer" aria-label="Supprimer la conversation">×</button>
        </li>`
      )
      .join("");

    // Clic sur une conversation → on la charge.
    els.sessionsList.querySelectorAll(".session-item").forEach((item) => {
      item.addEventListener("click", () => openSession(item.dataset.id));
    });

    // Clic sur la croix → on supprime (sans ouvrir la conversation).
    els.sessionsList.querySelectorAll(".session-delete").forEach((button) => {
      button.addEventListener("click", (event) => {
        event.stopPropagation();
        const id = button.closest(".session-item").dataset.id;
        deleteSession(id);
      });
    });
  } catch {
    // Silencieux : la liste des conversations n'est pas critique.
  }
}

/**
 * Supprime une conversation et tous ses messages.
 *
 * Si la conversation supprimée est celle actuellement ouverte, on revient
 * à l'écran d'accueil pour ne pas laisser l'utilisateur sur un fil vide.
 */
async function deleteSession(sessionId) {
  if (!confirm("Supprimer cette conversation et tous ses messages ?")) return;

  try {
    const response = await fetch(`/api/sessions/${sessionId}`, {
      method: "DELETE",
      headers: authHeaders(),
    });
    if (!response.ok) throw new Error("échec de la suppression");

    if (state.sessionId === sessionId) {
      state.sessionId = null;
      clearConversation();
    }

    toast("Conversation supprimée", "ok");
    await loadSessions();
  } catch (error) {
    toast(`Erreur : ${error.message}`, "err");
  }
}

/** Ouvre une conversation existante et charge ses derniers messages. */
async function openSession(sessionId) {
  state.sessionId = sessionId;
  state.history = [];
  state.loadedCount = 0;

  // On vide l'écran.
  const inner = document.querySelector(".chat-inner");
  if (inner) inner.remove();
  messageCount = 0;
  if (els.welcome) els.welcome.style.display = "none";

  document.querySelectorAll(".session-item").forEach((el) => {
    el.classList.toggle("active", el.dataset.id === sessionId);
  });

  await loadMessages(0);
}

/**
 * Charge une page de messages depuis la base.
 *
 * @param {number} offset  nombre de messages à sauter (pagination)
 */
async function loadMessages(offset) {
  try {
    const response = await fetch(
      `/api/sessions/${state.sessionId}/messages?limit=20&offset=${offset}`,
      { headers: authHeaders() }
    );
    if (!response.ok) return;

    const messages = await response.json();

    // S'il reste des messages plus anciens, on affiche le bouton "Voir plus".
    state.hasMore = messages.length === 20;
    els.loadMoreBtn.hidden = !state.hasMore;

    // Les messages arrivent du plus ancien au plus récent.
    // On les insère AVANT les messages déjà affichés (chargement par le haut).
    const inner = getOrCreateInner();
    const anchor = inner.firstChild;

    for (const m of messages) {
      const role = m.role === "user" ? "user" : "bot";
      const name = m.role === "user" ? "Vous" : "Atlas";
      const node = buildMessageNode(role, name, m.content, m.sources);
      inner.insertBefore(node, anchor);
      messageCount++;
    }

    state.loadedCount += messages.length;
    updateTopbar();

    // On garde l'historique local synchronisé (utile si l'utilisateur
    // se déconnecte en cours de route).
    state.history = messages.map((m) => [m.role, m.content]).slice(-20);

    if (offset === 0) scrollToBottom();
  } catch {
    // Silencieux.
  }
}

/** Récupère (ou crée) le conteneur interne du fil de discussion. */
function getOrCreateInner() {
  let inner = document.querySelector(".chat-inner");
  if (!inner) {
    inner = document.createElement("div");
    inner.className = "chat-inner";
    els.chat.appendChild(inner);
  }
  return inner;
}

/**
 * Construit un nœud de message SANS l'ajouter au DOM.
 * Utilisé pour insérer des messages anciens en haut du fil.
 */
function buildMessageNode(role, name, text, sources) {
  const wrapper = document.createElement("div");
  wrapper.className = `msg ${role}`;
  wrapper.innerHTML = `
    <div class="avatar">${role === "user" ? ICON_USER : ICON_BOT}</div>
    <div class="bubble">
      <div class="bubble-name">${name}</div>
      <div class="bubble-text">${renderMarkdown(text)}</div>
    </div>
  `;

  if (sources && sources.length) {
    const bubble = wrapper.querySelector(".bubble");
    const container = document.createElement("div");
    container.className = "sources";
    container.innerHTML =
      '<span class="sources-label">Sources</span>' +
      sources
        .map(
          (s) =>
            `<span class="source-chip">${ICON_FILE}${escapeHtml(
              String(s).replace(/\.(md|txt|pdf)$/i, "")
            )}</span>`
        )
        .join("");
    bubble.appendChild(container);
  }

  return wrapper;
}

/* ============================ Événements =============================== */

els.send.addEventListener("click", () => {
  const question = els.input.value.trim();
  if (!question) return;
  els.input.value = "";
  els.input.style.height = "auto";
  sendQuestion(question);
});

els.stop.addEventListener("click", () => {
  if (abortController) abortController.abort();
});

els.input.addEventListener("keydown", (event) => {
  // Entrée = envoyer ; Maj+Entrée = retour à la ligne.
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    els.send.click();
  }
});

els.input.addEventListener("input", () => {
  // La zone de saisie grandit avec le texte (jusqu'à une limite CSS).
  els.input.style.height = "auto";
  els.input.style.height = Math.min(els.input.scrollHeight, 168) + "px";
});

document.querySelectorAll(".suggestion").forEach((button) => {
  button.addEventListener("click", () => sendQuestion(button.textContent.trim()));
});

els.uploadBtn.addEventListener("click", () => els.fileInput.click());
els.fileInput.addEventListener("change", (event) => uploadFile(event.target.files[0]));
els.clearBtn.addEventListener("click", clearConversation);

// --- Authentification ---
els.authForm.addEventListener("submit", submitAuth);
els.tabLogin.addEventListener("click", () => setAuthMode("login"));
els.tabSignup.addEventListener("click", () => setAuthMode("signup"));
els.anonBtn.addEventListener("click", () => {
  state.token = null;
  state.email = null;
  enterApp();
});
els.anonLoginBtn.addEventListener("click", showAuthScreen);
els.logoutBtn.addEventListener("click", logout);
els.switchAccountBtn.addEventListener("click", showAuthScreen);

// --- Conversations ---
els.newSessionBtn.addEventListener("click", startNewSession);
els.loadMoreBtn.addEventListener("click", () => loadMessages(state.loadedCount));

/* ============================ Démarrage ================================ */

/**
 * Point d'entrée : restaure la session si le token est encore valide,
 * sinon affiche l'écran de connexion.
 */
async function boot() {
  if (state.token && (await validateToken())) {
    await enterApp();
  } else {
    showAuthScreen();
  }
}

boot();
