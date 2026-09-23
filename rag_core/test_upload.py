"""Test d'upload : vérifie qu'un upload fonctionne et n'efface pas les
autres documents de l'utilisateur."""

import json
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:8000"


def call(path, method="GET", body=None, token=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(BASE + path, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode()


def upload(path, content, token):
    boundary = "----AtlasBoundary"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{path}"\r\n'
        f"Content-Type: text/plain\r\n\r\n"
        f"{content}\r\n"
        f"--{boundary}--\r\n"
    ).encode()
    req = urllib.request.Request(BASE + "/api/upload", data=body, method="POST")
    req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode()


# --- Connexion ---
status, payload = call(
    "/api/auth/login", "POST",
    {"email": "isolation.test@atlas.dev", "password": "TestIsolation123"},
)
if status != 200:
    status, payload = call(
        "/api/auth/signup", "POST",
        {"email": "isolation.test@atlas.dev", "password": "TestIsolation123"},
    )
token = payload["token"]
print(f"OK  authentifie (status {status})")

# --- Upload de deux fichiers ---
status, r1 = upload("test_alpha.txt", "Le service client repond en 48 heures.", token)
print(f"UPLOAD 1 (status {status}) : {r1}")

status, r2 = upload("test_beta.txt", "La livraison express prend 24 heures.", token)
print(f"UPLOAD 2 (status {status}) : {r2}")

# --- Verification : les deux doivent etre presents ---
status, docs = call("/api/documents", token=token)
print(f"\nDOCUMENTS ({len(docs)}) :")
for doc in docs:
    print(f"  - {doc['name']} ({doc['chunks_count']} chunks)")

names = {d["name"] for d in docs}
if "test_alpha.txt" in names and "test_beta.txt" in names:
    print("\n=== UPLOAD VALIDE : les deux fichiers coexistent ===")
else:
    print(f"\nECHEC : fichiers manquants. presents = {names}")
