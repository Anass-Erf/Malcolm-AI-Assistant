import os
import argparse
import ssl
import hashlib
from datetime import datetime, timedelta
from typing import Dict, Any, List

import numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from opensearchpy import OpenSearch
from opensearchpy.helpers import scan

from vectorstore import FAISSVectorStore
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
# -----------------------------
# Config .env
# -----------------------------
load_dotenv()

OPENSEARCH_URL = os.getenv("OPENSEARCH_URL", "https://localhost:9200")
OPENSEARCH_USER = os.getenv("OPENSEARCH_USER", "admin")
OPENSEARCH_PASS = os.getenv("OPENSEARCH_PASS", "changeme")
VERIFY_SSL = os.getenv("VERIFY_SSL", "false").lower() == "true"

VECTOR_INDEX_PATH = os.getenv("VECTOR_INDEX_PATH", "./data/vectors.faiss")
VECTOR_PAYLOAD_PATH = os.getenv("VECTOR_PAYLOAD_PATH", "./data/payload.pkl")
EMBEDDINGS_MODEL = os.getenv("EMBEDDINGS_MODEL", "BAAI/bge-m3")

# Indices à parcourir
INDICES = [
    "malcolm_beats_zeek_*",       # Tous les logs Zeek
    "malcolm_beats_suricata_*",   # Toutes les alertes Suricata
    "arkime_sessions3-*",         # Sessions Arkime (optionnel mais utile)
]

# -----------------------------
# OpenSearch client
# -----------------------------
def os_client() -> OpenSearch:
    if not VERIFY_SSL:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
    else:
        ctx = None

    return OpenSearch(
        hosts=[OPENSEARCH_URL],
        http_auth=(OPENSEARCH_USER, OPENSEARCH_PASS),
        use_ssl=OPENSEARCH_URL.startswith("https"),
        verify_certs=VERIFY_SSL,
        ssl_context=ctx,
        timeout=60,
        max_retries=3,
        retry_on_timeout=True,
    )

# -----------------------------
# Helpers
# -----------------------------
def get_first(src: Dict[str, Any], *keys):
    """Retourne la première valeur non vide en naviguant avec des clés 'a.b.c'."""
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


def stable_int64_id(index_name: str, os_id: str) -> int:
    h = hashlib.sha1(f"{index_name}:{os_id}".encode("utf-8")).digest()
    # Interprétation signée -> résultat dans [-2^63, 2^63-1]
    return int.from_bytes(h[:8], "big", signed=True)



# -----------------------------
# Flatten doc -> texte
# -----------------------------
def doc_to_text(doc: Dict[str, Any]) -> str:
    """
    Aplatis un événement Malcolm (Zeek/Suricata/Arkime) en texte concis pour embedding.
    Gère les clés alternatives (fallbacks).
    """
    _s = doc.get("_source", {})
    idx = doc.get("_index", "unknown")
    parts = []

    # Horodatage (plusieurs variantes selon la source)
    ts = get_first(_s, "@timestamp", "timestamp", "ts", "firstPacket", "lastPacket")
    if ts: parts.append(f"[{idx}] @ {ts}")
    else:  parts.append(f"[{idx}]")

    # Tuple réseau (Suricata/Zeek/ECS/Arkime)
    src_ip   = get_first(_s, "src_ip", "id.orig_h", "source.ip", "srcIp")
    src_port = get_first(_s, "src_port", "id.orig_p", "source.port", "srcPort")
    dst_ip   = get_first(_s, "dest_ip", "id.resp_h", "destination.ip", "dstIp")
    dst_port = get_first(_s, "dest_port", "id.resp_p", "destination.port", "dstPort")
    proto    = get_first(_s, "proto", "network.transport", "protocol")

    if src_ip:   parts.append(f"src={src_ip}")
    if src_port: parts.append(f"sport={src_port}")
    if dst_ip:   parts.append(f"dst={dst_ip}")
    if dst_port: parts.append(f"dport={dst_port}")
    if proto:    parts.append(f"proto={proto}")

    # Suricata (alertes)
    sig = get_first(_s, "alert.signature")
    cat = get_first(_s, "alert.category")
    sev = get_first(_s, "alert.severity", "severity")
    if sig: parts.append(f"sig='{sig}'")
    if cat: parts.append(f"cat={cat}")
    if sev is not None: parts.append(f"sev={sev}")

    # DNS
    dns_q = get_first(_s, "dns.qry_name", "dns.query")
    if dns_q: parts.append(f"dns={dns_q}")

    # HTTP
    http_host = get_first(_s, "http.host")
    http_uri  = get_first(_s, "http.uri")
    http_mtd  = get_first(_s, "http.method")
    http_ua   = get_first(_s, "http.user_agent")
    http_sc   = get_first(_s, "http.status_code")
    if http_host: parts.append(f"http.host={http_host}")
    if http_uri:  parts.append(f"http.uri={http_uri}")
    if http_mtd:  parts.append(f"http.method={http_mtd}")
    if http_ua:   parts.append(f"http.ua={http_ua}")
    if http_sc:   parts.append(f"http.code={http_sc}")

    # TLS/SSL
    sni = get_first(_s, "tls.server_name", "ssl.server_name")
    if sni: parts.append(f"sni={sni}")
    tls_ver = get_first(_s, "tls.version", "ssl.version")
    if tls_ver: parts.append(f"tls={tls_ver}")
    tls_subj = get_first(_s, "ssl.subject")
    tls_iss  = get_first(_s, "ssl.issuer")
    if tls_subj: parts.append(f"ssl.subj={tls_subj}")
    if tls_iss:  parts.append(f"ssl.iss={tls_iss}")

    # Identifiants & métriques
    for k in [
        "community_id", "flow_id", "uid",
        "service", "duration", "bytes", "raw_bytes", "packets", "orig_bytes", "resp_bytes",
        "ja3", "ja3s",
        "network.direction", "event.type", "event.category"
    ]:
        v = get_first(_s, k)
        if v not in (None, "", []):
            parts.append(f"{k}={v}")

    # Ajoute quelques champs top-level simples pour enrichir (limite 20)
    keep = 0
    for k, v in _s.items():
        if keep >= 20:
            break
        if isinstance(v, (str, int, float, bool)) and k not in ["@timestamp", "timestamp", "ts"]:
            parts.append(f"{k}={v}")
            keep += 1

    return " | ".join(parts)

# -----------------------------
# Génération de docs (avec bool should sur plusieurs champs temps)
# -----------------------------
def build_time_query(gte_iso: str) -> Dict[str, Any]:
    """Filtre par date en acceptant plusieurs champs (@timestamp/timestamp/ts/firstPacket)."""
    return {
        "query": {
            "bool": {
                "should": [
                    {"range": {"@timestamp": {"gte": gte_iso}}},
                    {"range": {"timestamp": {"gte": gte_iso}}},
                    {"range": {"ts": {"gte": gte_iso}}},
                    {"range": {"firstPacket": {"gte": gte_iso}}},
                ],
                "minimum_should_match": 1
            }
        }
    }

def generate_docs(client: OpenSearch, since: timedelta):
    gte_iso = (datetime.utcnow() - since).isoformat() + "Z"
    query = build_time_query(gte_iso)
    for index in INDICES:
        try:
            for hit in scan(client, index=index, query=query, size=1000):
                yield hit
        except Exception as e:
            print(f"[WARN] skipping index {index}: {e}")

# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=7, help="How many days back to index")
    parser.add_argument("--batch-size", type=int, default=5000, help="Upsert batch size")
    parser.add_argument("--device", default=None, help='Force device for embeddings (e.g. "cpu" or "cuda")')
    args = parser.parse_args()

    client = os_client()

    # Embeddings
    print("[*] Loading embeddings model:", EMBEDDINGS_MODEL)

    model = SentenceTransformer(EMBEDDINGS_MODEL, device=args.device) if args.device \
        else SentenceTransformer(EMBEDDINGS_MODEL)
    dim = model.get_sentence_embedding_dimension()
    
    # Vector store
    store = FAISSVectorStore(dim=dim, index_path=VECTOR_INDEX_PATH, payload_path=VECTOR_PAYLOAD_PATH)

    texts: List[str] = []
    ids: List[int] = []
    payloads: List[Dict[str, Any]] = []

    since = timedelta(days=args.days)
    batch = 0
    n_docs = 0

    for doc in generate_docs(client, since):
        text = doc_to_text(doc)

        # ID stable basé sur (index, _id) pour éviter les doublons
        idx_name = doc.get("_index", "unknown")
        os_id = doc.get("_id", "")
        sid = stable_int64_id(idx_name, os_id)

        texts.append(text)
        ids.append(sid)
        payloads.append({"_index": idx_name, "_id": os_id, "text": text})
        n_docs += 1

        if len(texts) >= args.batch_size:
            X = model.encode(texts, normalize_embeddings=True).astype("float32")
            store.upsert(X, np.array(ids, dtype="int64"), payloads)
            store.save()
            print(f"[+] Upserted batch {batch} of {len(texts)} docs (total so far: {n_docs})")
            batch += 1
            texts, ids, payloads = [], [], []

    if texts:
        X = model.encode(texts, normalize_embeddings=True).astype("float32")
        store.upsert(X, np.array(ids, dtype="int64"), payloads)
        store.save()
        print(f"[+] Upserted final batch of {len(texts)} docs (total: {n_docs})")

    print("[OK] Index build/update complete.")

if __name__ == "__main__":
    main()
