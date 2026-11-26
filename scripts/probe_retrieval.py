"""Retrieval probe for finding F7 (exact CVE ID retrieval). Local only: no LLM calls.

Uses chatbot.kb.KB.query, the same call Copilot.answer makes. To guarantee data/chroma is
not touched, the collection is copied to a temp dir and the copy is opened.
Chroma 'cosine' space returns distance = 1 - cosine_similarity; KB.query converts
explicitly (similarity = 1.0 - distance). Both are recorded.
"""
import json, shutil, sys, tempfile, re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from chatbot import kb as kbmod                      # noqa: E402
from chatbot.run_redteam import BENIGN_QUESTIONS     # noqa: E402

TOP_K = 5
ID_RE = re.compile(r"CVE-\d{4}-\d{4,}|T\d{4}(?:\.\d{3})?")

tmp = Path(tempfile.mkdtemp(prefix="chroma_copy_"))
shutil.copytree(kbmod.CHROMA_DIR, tmp / "chroma")
kb = kbmod.KB(persist_dir=tmp / "chroma")


def doc_id_of(meta):
    if meta.get("source") == "cisa-kev":
        return f"kev-{meta['cve_id']}"
    return f"attack-{meta.get('technique_id', '')}"


def probe(question, target):
    hits = kb.query(question, k=TOP_K)
    ids = [doc_id_of(h["meta"]) for h in hits]
    sims = [h["similarity"] for h in hits]
    dists = [h["distance"] for h in hits]
    rank = ids.index(target) + 1 if target in ids else None
    # target score even when outside top-k: query the target doc's stored embedding
    tgt_sim = None
    if rank:
        tgt_sim = sims[rank - 1]
    else:
        got = kb.coll.get(ids=[target], include=["embeddings"])
        if len(got["ids"]):
            import numpy as np
            q = kb.embedder.encode([question])[0]
            e = np.array(got["embeddings"][0])
            tgt_sim = float(np.dot(q, e) / (np.linalg.norm(q) * np.linalg.norm(e)))
    return {"question": question, "target_id": target, "rank_in_top5": rank,
            "in_top5": rank is not None, "target_similarity": tgt_sim,
            "top5_ids": ids, "top5_similarities": sims, "top5_distances": dists,
            "cutoff_5th_similarity": sims[-1]}


out = {"top_k": TOP_K,
       "score_conversion": "chroma hnsw:space=cosine returns distance = 1 - cosine_similarity; "
                           "similarity = 1.0 - distance (KB.query). 'cosine' in the report = similarity.",
       "collection_count": kb.count()}

# a. the CVE-2024-3400 benign question
q3 = next(q for q in BENIGN_QUESTIONS if "CVE-2024-3400" in q)
out["a_cve_2024_3400"] = probe(q3, "kev-CVE-2024-3400")

# b1. benign questions naming an exact ID
b1 = []
for q in BENIGN_QUESTIONS:
    ids = ID_RE.findall(q)
    if ids:
        tid = ids[0]
        target = f"kev-{tid}" if tid.startswith("CVE") else f"attack-{tid}"
        b1.append(probe(q, target))
out["b1_benign_questions_naming_ids"] = {
    "n_benign_questions": len(BENIGN_QUESTIONS), "n_naming_an_id": len(b1),
    "n_exact_doc_in_top5": sum(r["in_top5"] for r in b1), "probes": b1}

# b2. fixed probe: first 5 KEV doc ids sorted
# data/kev is not on disk, so KEV ids come from the stored collection (source=cisa-kev)
kev_ids = sorted(kb.coll.get(where={"source": "cisa-kev"}, include=[])["ids"])
chosen = kev_ids[:5]
b2 = [probe(f"What is CISA's required action for {t[4:]}?", t) for t in chosen]
out["b2_fixed_kev_probe"] = {"selection": "first 5 KEV doc ids, sorted ascending",
    "template": "What is CISA's required action for <CVE>?", "targets": chosen,
    "n_exact_doc_in_top5": sum(r["in_top5"] for r in b2), "n_probes": len(b2), "probes": b2}

# c. requiredAction absent from embedded text
src = (ROOT / "chatbot" / "kb.py").read_text()
fn = src[src.index("def load_kev_docs"):src.index("def get_embedder")]
doc_id = "kev-CVE-2024-3400"
stored = kb.coll.get(ids=[doc_id], include=["documents"])["documents"][0]
kev_json = json.loads(kbmod.KEV_JSON.read_text()) if kbmod.KEV_JSON.exists() else None  # absent in this checkout
ra = None
if kev_json:
    ra = next((v.get("requiredAction") for v in kev_json["vulnerabilities"]
               if v.get("cveID") == "CVE-2024-3400"), None)
out["c_required_action_absent"] = {
    "doc_id_checked": doc_id,
    "load_kev_docs_source_references_requiredAction_in_text_bits": "requiredAction" in fn.split("bits = [")[1].split("]")[0],
    "stored_text": stored,
    "stored_text_contains_requiredAction_word": "requiredAction" in stored,
    "required_action_value_in_kev_json": ra,
    "stored_text_contains_required_action_value": bool(ra) and ra in stored,
    "required_action_absent_from_embedded_text": not (bool(ra) and ra in stored) and "requiredAction" not in stored,
}
dst = ROOT / "results" / "task2_retrieval_probe.json"
dst.write_text(json.dumps(out, indent=2))
shutil.rmtree(tmp, ignore_errors=True)
print("wrote", dst)
