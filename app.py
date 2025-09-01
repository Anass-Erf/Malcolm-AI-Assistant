import os
import ssl
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

import numpy as np
import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from opensearchpy import OpenSearch
from opensearchpy.helpers import scan

from vectorstore import FAISSVectorStore

load_dotenv()

# ---- Config ----
OPENSEARCH_URL = os.getenv("OPENSEARCH_URL", "https://localhost:9200")
OPENSEARCH_USER = os.getenv("OPENSEARCH_USER", "admin")
OPENSEARCH_PASS = os.getenv("OPENSEARCH_PASS", "changeme")
VERIFY_SSL = os.getenv("VERIFY_SSL", "false").lower() == "true"

VECTOR_INDEX_PATH = os.getenv("VECTOR_INDEX_PATH", "./data/vectors.faiss")
VECTOR_PAYLOAD_PATH = os.getenv("VECTOR_PAYLOAD_PATH", "./data/payload.pkl")
EMBEDDINGS_MODEL = os.getenv("EMBEDDINGS_MODEL", "BAAI/bge-m3")

OLLAMA_BASE = os.getenv("OLLAMA_BASE", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")

# Malcolm / Arkime index patterns
RECENT_INDEX_PATTERNS = [
    "malcolm_beats_zeek_*",
    "malcolm_beats_suricata_*",
    "arkime_sessions3-*",
]

# ---- FastAPI ----
app = FastAPI(title="Local SOC AI (FR)", version="0.1.1")

# ---- OpenSearch client ----
if not VERIFY_SSL:
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
else:
    ctx = None

os_client = OpenSearch(
    hosts=[OPENSEARCH_URL],
    http_auth=(OPENSEARCH_USER, OPENSEARCH_PASS),
    use_ssl=OPENSEARCH_URL.startswith("https"),
    verify_certs=VERIFY_SSL,
    ssl_context=ctx,
    timeout=60,
    max_retries=3,
    retry_on_timeout=True,
)

# ---- Embeddings / Vector store ----
embedder = SentenceTransformer(EMBEDDINGS_MODEL)
# Faster on 1070 and often enough for logs
embedder.max_seq_length = 256

dim = embedder.get_sentence_embedding_dimension()
store = FAISSVectorStore(dim=dim, index_path=VECTOR_INDEX_PATH, payload_path=VECTOR_PAYLOAD_PATH)

# ---- Schemas ----
class AskRequest(BaseModel):
    question: str = Field(..., min_length=3)
    days: Optional[int] = Field(7, ge=0, le=90, description="Fenêtre temporelle pour les docs frais")
    top_k: Optional[int] = Field(20, ge=1, le=50, description="Nombre de passages vectoriels")
    want_chart: Optional[bool] = True

class AskResponse(BaseModel):
    answer: str
    used_sources: List[Dict[str, Any]]
    chart: Optional[Dict[str, Any]] = None

# ---- Helpers ----
def build_time_query(gte_iso: str) -> Dict[str, Any]:
    """Accept multiple time fields to match Malcolm/Arkime sources."""
    return {
        "query": {
            "bool": {
                "should": [
                    {"range": {"@timestamp":  {"gte": gte_iso}}},
                    {"range": {"timestamp":   {"gte": gte_iso}}},
                    {"range": {"ts":          {"gte": gte_iso}}},
                    {"range": {"firstPacket": {"gte": gte_iso}}},
                    {"range": {"lastPacket":  {"gte": gte_iso}}},
                ],
                "minimum_should_match": 1
            }
        },
        "size": 1000
    }

def os_recent_docs(days: int, limit: int = 2000) -> List[Dict[str, Any]]:
    gte_iso = (datetime.utcnow() - timedelta(days=days)).isoformat() + "Z"
    q = build_time_query(gte_iso)
    docs: List[Dict[str, Any]] = []
    for idx in RECENT_INDEX_PATTERNS:
        try:
            for hit in scan(os_client, index=idx, query=q, size=1000):
                docs.append(hit)
                if len(docs) >= limit:
                    return docs
        except Exception:
            continue
    return docs

async def ollama_complete(prompt: str) -> str:
    # Non-streaming call (explicit)
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(
            f"{OLLAMA_BASE}/api/generate",
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
        )
        r.raise_for_status()
        data = r.json()
        # Ollama returns {"response": "...", "done": true, ...}
        txt = data.get("response") or data.get("data") or ""
        if not txt:
            raise HTTPException(status_code=502, detail="Réponse vide d'Ollama")
        return txt

def get_first(src: Dict[str, Any], *keys):
    cur: Any
    for k in keys:
        cur = src
        ok = True
        for part in k.split("."):
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                ok = False
                break
        if ok and cur not in (None, "", []):
            return cur
    return None

def result_to_text(doc: Dict[str, Any]) -> str:
    _s = doc.get("_source", {})
    ts = get_first(_s, "@timestamp", "timestamp", "ts", "firstPacket", "lastPacket") or "-"
    src = get_first(_s, "src_ip", "id.orig_h", "source.ip", "srcIp") or "-"
    dst = get_first(_s, "dest_ip", "id.resp_h", "destination.ip", "dstIp") or "-"
    proto = get_first(_s, "proto", "network.transport", "protocol") or "-"
    sig = get_first(_s, "alert.signature") or "-"
    host = get_first(_s, "http.host") or "-"
    return f"{ts} src={src} dst={dst} proto={proto} sig={sig} host={host}"

def make_chart_from_docs(docs: List[Dict[str, Any]]) -> Dict[str, Any]:
    from collections import Counter
    c = Counter()
    for d in docs:
        s = d.get("_source", {})
        ip = get_first(s, "src_ip", "id.orig_h", "source.ip", "srcIp") or "unknown"
        c[ip] += 1
    top = c.most_common(10)
    labels = [k for k, _ in top]
    values = [v for _, v in top]
    return {
        "type": "bar",
        "title": "Top émetteurs (comptage d'événements)",
        "x": labels,
        "y": values,
        "x_title": "IP source",
        "y_title": "Nombre d'événements"
    }

def build_prompt_fr(question: str, contexts: List[str], findings: List[str]) -> str:
    ctx_txt = "\n\n".join(f"- {c}" for c in contexts[:20])
    res_txt = "\n".join(f"- {f}" for f in findings[:20])
    return (
        "Tu es un analyste SOC francophone. Tu disposes des journaux réseau de Malcolm (Zeek/Suricata/Arkime).\n"
        "Réponds au format : Constat → Analyse → Impact → Recommandations (actions concrètes). Cite la période.\n\n"
        f"Question : {question}\n\n"
        "Contexte (extraits pertinents) :\n"
        f"{ctx_txt}\n\n"
        "Constats chiffrés récents :\n"
        f"{res_txt}\n\n"
        "Réponds en français clair, avec commandes ou règles quand utile.\n"
    )

def encode_query(texts: List[str]) -> np.ndarray:
    # Convert-to-numpy avoids slow .cpu() later; tune batch_size for your GPU
    return embedder.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        batch_size=64,
        show_progress_bar=False
    ).astype("float32")

# ---- Routes ----
@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    question = req.question.strip()
    if len(question) < 3:
        raise HTTPException(status_code=400, detail="Question vide ou trop courte.")

    # 1) Vector context from FAISS
    contexts: List[str] = []
    hits_list = []
    if store.count() > 0:
        q_vec = encode_query([question])
        hits_list = store.search(q_vec, top_k=req.top_k or 20)[0]
        contexts = [h.payload.get("text", "") for h in hits_list]
    else:
        # No vectors yet; continue with OS-only signal
        contexts = []

    # 2) Fresh docs from OpenSearch (time-bounded)
    fresh_docs = os_recent_docs(days=req.days or 7, limit=1000)
    findings = [result_to_text(d) for d in fresh_docs[:50]]

    # 3) LLM prompt → Ollama
    prompt = build_prompt_fr(question, contexts, findings)
    try:
        answer = await ollama_complete(prompt)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ollama error: {e}")

    # 4) Optional chart
    chart = make_chart_from_docs(fresh_docs) if req.want_chart else None

    used_sources = [{"vector_hit_score": h.score, **h.payload} for h in hits_list[:10]]
    return AskResponse(answer=answer, used_sources=used_sources, chart=chart)

@app.get("/healthz")
async def health():
    return {"ok": True, "vectors": store.count()}
