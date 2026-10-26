import json, re, sys
from rank_bm25 import BM25Okapi

D = json.load(open("data.json", encoding="utf-8"))
defs, qs = D["definitions"], D["queries"]

# تطبيع بسيط: حذف التشكيل والتطويل، توحيد الألف والياء والتاء المربوطة
def norm(s):
    s = re.sub(r"[ً-ْـ]", "", s)
    s = re.sub("[إأآ]", "ا", s)
    s = s.replace("ى", "ي").replace("ة", "ه")
    return s

STOP = set("في من الذي التي الى على عن ان او ما هو هي يقال اي كل".split())
def tok(s):
    return [t for t in re.findall(r"\w+", norm(s)) if t not in STOP]

# نصّ التعريف دون الكلمة المعرَّفة (حتى لا يكون الجواب مكشوفًا)
def body(d):
    return d["text"].split(":", 1)[1]

def rank_of(scores, ans_idx):
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    return order.index(ans_idx) + 1

ids = [d["id"] for d in defs]
bm = BM25Okapi([tok(body(d)) for d in defs])
res = {"BM25": []}
for q in qs:
    s = bm.get_scores(tok(q["text"]))
    res["BM25"].append(rank_of(list(s), ids.index(q["answer"])))

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    name = sys.argv[1] if len(sys.argv) > 1 else "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    m = SentenceTransformer(name)
    E = m.encode([body(d) for d in defs], normalize_embeddings=True)
    Q = m.encode([q["text"] for q in qs], normalize_embeddings=True)
    S = Q @ E.T
    res[name.split("/")[-1]] = [rank_of(list(S[i]), ids.index(q["answer"])) for i, q in enumerate(qs)]
except Exception as e:
    print("SEMANTIC FAILED:", type(e).__name__, str(e)[:300])

print("\nرتبة الجواب الصحيح لكل سؤال (1 = الأفضل)")
head = "سؤال | " + " | ".join(res)
print(head)
for i, q in enumerate(qs):
    print(q["id"], "|", " | ".join(str(res[k][i]) for k in res))
print()
for k, r in res.items():
    mrr = sum(1 / x for x in r) / len(r)
    print(f"{k}: MRR={mrr:.2f}  top1={sum(x==1 for x in r)}/{len(r)}  top3={sum(x<=3 for x in r)}/{len(r)}")
json.dump(res, open("results.json", "w"), ensure_ascii=False, indent=1)
