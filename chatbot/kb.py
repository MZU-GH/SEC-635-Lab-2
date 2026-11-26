"""Build the Task 2 knowledge base: ATT&CK techniques + CISA KEV, in chromadb.

The lab brief points at "the RAG copilot of Chapter 4, Activity 4.4" for the
architecture. That activity does not exist - Chapter 4 ships no RAG code, no embedding
model, no chunking strategy and no system prompt (verified by full read; see
report/REPRODUCING.md section 14). So every choice here is ours and is stated explicitly:

  embedding model  all-MiniLM-L6-v2 (384-dim). Chapter 4 names none; this is the
                   sentence-transformers default and runs on CPU, which matters because
                   the corpus is ~2,500 documents and is embedded once.
  chunking         one document per live technique, no splitting. ATT&CK descriptions
                   are 0.2-8 KB; splitting them would scatter a technique's semantics
                   and, for Task 2A, would give an attacker more retrieved fragments to
                   hide a payload in. Whole-technique docs are the more conservative
                   choice and the one we attack.
  top-k            5, cosine distance.
  deprecation      drop `revoked` OR `x_mitre_deprecated`. STIX has no `deprecated`
                   key; filtering on that silently keeps 161 dead techniques.

The corpus is also the attack surface for Task 2A's second half: `add_document()` is
the deliberate poisoning hook the attacker uses, and it is the same code path a
legitimate ingest would use, which is the point.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

ATTACK_JSON = ROOT / "data" / "attack" / "enterprise-attack.json"
KEV_JSON = ROOT / "data" / "kev" / "known_exploited_vulnerabilities.json"

EMBED_MODEL_DIR = ROOT / "models" / "all-MiniLM-L6-v2"
CHROMA_DIR = ROOT / "data" / "chroma"
COLLECTION = "sec635_kb"

_HTML_TAG = re.compile(r"</?code>|</?a>|</?em>|</?strong>")
_CITATION = re.compile(r"\(Citation: [^)]*\)")


def clean(text: str) -> str:
    """Strip the markup ATT&CK descriptions carry.

    `description` fields contain literal <code>...</code> and (Citation: ...) markers.
    Stripping them is a real decision, not cosmetics: leaving them in means an attacker
    who can write a document can also use angle-bracket markup that looks structurally
    like our own <doc> delimiters. We strip on ingest AND the guard strips at query time.
    """
    text = _HTML_TAG.sub("", text or "")
    text = _CITATION.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


def technique_id(obj: dict) -> str:
    for ref in obj.get("external_references") or []:
        eid = ref.get("external_id", "")
        if eid.startswith("T"):
            return eid
    return ""


def load_attack_docs(path: Path = ATTACK_JSON) -> list[dict]:
    """One document per live enterprise-attack technique, parents and sub-techniques."""
    bundle = json.loads(path.read_text())
    docs = []
    for obj in bundle.get("objects", []):
        if obj.get("type") != "attack-pattern":
            continue
        if obj.get("revoked") or obj.get("x_mitre_deprecated"):
            continue
        tid = technique_id(obj)
        if not tid:
            continue
        domains = obj.get("x_mitre_domains") or []
        if domains and "enterprise-attack" not in domains:
            continue
        phases = [p.get("phase_name", "") for p in (obj.get("kill_chain_phases") or [])]
        body = clean(obj.get("description", ""))
        docs.append({
            "id": f"attack-{tid}",
            "text": f"{tid} {obj.get('name', '')}: {body}",
            "meta": {
                "source": "mitre-attack",
                "technique_id": tid,
                "name": obj.get("name", ""),
                "phases": ",".join(phases),
                "platforms": ",".join(obj.get("x_mitre_platforms") or []),
                "is_subtechnique": "." in tid,
                "kind": "technique",
            },
        })
    return docs


def load_kev_docs(path: Path = KEV_JSON) -> list[dict]:
    """One document per KEV entry.

    Only `shortDescription` is embedded. `requiredAction` and `notes` are boilerplate
    and URL soup that would dominate the embedding space and dilute retrieval.
    """
    kev = json.loads(path.read_text())
    docs = []
    for i, v in enumerate(kev.get("vulnerabilities", [])):
        bits = [
            f"{v.get('cveID', '')}: {v.get('vulnerabilityName', '')}",
            f"{v.get('vendorProject', '')} {v.get('product', '')}.",
            clean(v.get("shortDescription", "")),
            f"Known ransomware use: {v.get('knownRansomwareCampaignUse', 'No')}.",
            f"Added {v.get('dateAdded', '')}, remediation due {v.get('dueDate', '')}.",
        ]
        docs.append({
            "id": f"kev-{v.get('cveID', i)}",
            "text": " ".join(b for b in bits if b.strip()),
            "meta": {
                "source": "cisa-kev",
                "cve_id": v.get("cveID", ""),
                "vendor": v.get("vendorProject", ""),
                "product": v.get("product", ""),
                "date_added": v.get("dateAdded", ""),
                "ransomware": v.get("knownRansomwareCampaignUse", "No"),
                "kind": "vulnerability",
            },
        })
    return docs


def get_embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(str(EMBED_MODEL_DIR), local_files_only=True)


class KB:
    """chromadb collection with a local embedder.

    `add_document` is public on purpose. Task 2A plants a poisoned document through
    exactly this call, which is the same one an analyst adding an internal runbook
    would use - the attack is "someone with ingest access writes a document", not a
    bug in our vector store.
    """

    def __init__(self, persist_dir: Path = CHROMA_DIR, rebuild: bool = False) -> None:
        import chromadb
        self.embedder = get_embedder()
        self.client = chromadb.PersistentClient(path=str(persist_dir))
        args = {"name": COLLECTION, "metadata": {"hnsw:space": "cosine"}}
        if rebuild:
            try:
                self.client.delete_collection(COLLECTION)
            except Exception:
                pass
        self.coll = self.client.get_or_create_collection(**args)

    def ingest(self, docs: list[dict]) -> int:
        if not docs:
            return 0
        B = 256
        for i in range(0, len(docs), B):
            chunk = docs[i:i + B]
            self.coll.upsert(
                ids=[d["id"] for d in chunk],
                documents=[d["text"] for d in chunk],
                metadatas=[d["meta"] for d in chunk],
                embeddings=self.embedder.encode([d["text"] for d in chunk],
                                                show_progress_bar=False).tolist(),
            )
        return len(docs)

    def add_document(self, doc_id: str, text: str, meta: dict | None = None) -> None:
        """The poisoning entry point. Identical path to a legitimate ingest."""
        m = {"source": "internal-runbook", "kind": "injected", "name": ""}
        m.update(meta or {})
        self.coll.upsert(
            ids=[doc_id], documents=[text], metadatas=[m],
            embeddings=self.embedder.encode([text]).tolist(),
        )

    def delete_document(self, doc_id: str) -> None:
        self.coll.delete(ids=[doc_id])

    def query(self, text: str, k: int = 5) -> list[dict]:
        res = self.coll.query(
            query_embeddings=self.embedder.encode([text]).tolist(),
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        out = []
        for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0],
                                   res["distances"][0]):
            out.append({"text": doc, "meta": meta, "distance": float(dist),
                        "similarity": float(1.0 - dist)})
        return out

    def count(self) -> int:
        return self.coll.count()


def build(rebuild: bool = True) -> KB:
    kb = KB(rebuild=rebuild)
    if kb.count() == 0 or rebuild:
        kb.ingest(load_attack_docs())
        kb.ingest(load_kev_docs())
    return kb


if __name__ == "__main__":
    attack = load_attack_docs()
    kev = load_kev_docs()
    print(f"ATT&CK live technique docs: {len(attack)} "
          f"(parents {sum(1 for d in attack if not d['meta']['is_subtechnique'])}, "
          f"sub-techniques {sum(1 for d in attack if d['meta']['is_subtechnique'])})")
    print(f"KEV docs: {len(kev)}")
    kb = build(rebuild=True)
    print(f"chroma collection size: {kb.count()}")
    for hit in kb.query("attacker scanning open ports on a host", k=3):
        print(f"  {hit['similarity']:.3f}  {hit['text'][:110]}")
