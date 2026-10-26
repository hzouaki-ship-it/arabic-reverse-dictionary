"""المعجم العكسي الدلالي العربي — نسخة تجريبية.
يكتب المستخدم وصفًا لمعنى، فيعرض التطبيق أقرب التعريفات والكلمات."""
import json, re
import gradio as gr
from rank_bm25 import BM25Okapi

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DATA = json.load(open("data.json", encoding="utf-8"))["definitions"]

def norm(s):
    s = re.sub(r"[ً-ْـ]", "", s)
    s = re.sub("[إأآ]", "ا", s)
    return s.replace("ى", "ي").replace("ة", "ه")

STOP = set("في من الذي التي الى على عن ان او ما هو هي يقال اي كل".split())
def tok(s):
    return [t for t in re.findall(r"\w+", norm(s)) if t not in STOP]

def body(d):
    return d["text"].split(":", 1)[1].strip()

BM = BM25Okapi([tok(body(d)) for d in DATA])
_model = {}

def semantic_scores(query):
    if "m" not in _model:
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer(MODEL_NAME)
        _model["m"] = m
        _model["E"] = m.encode([body(d) for d in DATA], normalize_embeddings=True)
    q = _model["m"].encode([query], normalize_embeddings=True)
    return list(map(float, (q @ _model["E"].T)[0]))

def search(query, method, k=5):
    query = (query or "").strip()
    if not query:
        return []
    if method == "بحث بالألفاظ (BM25)":
        scores = list(map(float, BM.get_scores(tok(query))))
    else:
        scores = semantic_scores(query)
    order = sorted(range(len(DATA)), key=lambda i: -scores[i])[:k]
    return [[r + 1, DATA[i]["word"], DATA[i]["sense"], body(DATA[i]), round(scores[i], 3)]
            for r, i in enumerate(order)]

with gr.Blocks(title="المعجم العكسي الدلالي") as demo:
    gr.Markdown("# المعجم العكسي الدلالي العربي\nاكتبي وصف المعنى الذي تحضر صورته في ذهنك ولا تتذكرين لفظه، "
                "وسيعرض التطبيق أقرب التعريفات إليه.\n\n*نسخة تجريبية: 30 تعريفًا فقط، وهي بيانات توضيحية.*")
    q = gr.Textbox(label="وصف المعنى", placeholder="مثال: شقيق الوالدة", rtl=True)
    m = gr.Radio(["بحث بالمعنى (نموذج دلالي)", "بحث بالألفاظ (BM25)"],
                 value="بحث بالمعنى (نموذج دلالي)", label="طريقة البحث")
    out = gr.Dataframe(headers=["الرتبة", "الكلمة", "المعنى", "التعريف", "الدرجة"], wrap=True)
    gr.Button("ابحث").click(search, [q, m], out)
    q.submit(search, [q, m], out)
    gr.Examples([["شقيق الوالدة"], ["الجزء الذي يعلو الجسد وفيه المخ"], ["فترة القحط التي يقل فيها المطر"]], q)

if __name__ == "__main__":
    demo.launch(share=True)
