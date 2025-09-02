import streamlit as st
import pandas as pd
import re, nltk, spacy, time, sqlite3, math
import arxiv
from sentence_transformers import SentenceTransformer
import faiss, numpy as np
from bertopic import BERTopic
from nltk.corpus import stopwords
from datetime import datetime

# ---------------------------
# SETUP
# ---------------------------
st.set_page_config(page_title="SmartResearch Advisor", layout="centered")
nltk.download('punkt')
nltk.download('stopwords')
STOP = set(stopwords.words("english"))
import en_core_web_sm
nlp = en_core_web_sm.load()
import torch
from sentence_transformers import SentenceTransformer

# Ensure torch uses CPU safely
torch.set_default_dtype(torch.float32)

# Load embeddings model on CPU
EMB_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")



# ---------------------------
# DATA FUNCTIONS
# ---------------------------
def fetch_arxiv(query="artificial intelligence", max_results=150):
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )
    rows = []
    for r in search.results():
        rows.append({
            "title": r.title,
            "abstract": r.summary,
            "url": r.entry_id,
            "published": r.published.date().isoformat()
        })
    return pd.DataFrame(rows)

def clean_text(t: str) -> str:
    t = re.sub(r"\s+", " ", t).strip()
    return t

def normalize(doc: str) -> str:
    d = nlp(doc)
    tokens = []
    for t in d:
        if t.is_stop or t.is_punct or t.like_num:
            continue
        lemma = t.lemma_.lower()
        if lemma and lemma not in STOP and len(lemma) > 2:
            tokens.append(lemma)
    return " ".join(tokens)

def prepare_corpus(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["text"] = (df["title"].fillna("") + ". " + df["abstract"].fillna("")).apply(clean_text)
    df["norm"] = df["text"].apply(normalize)
    return df

# ---------------------------
# EMBEDDINGS + FAISS
# ---------------------------
class VectorIndex:
    def __init__(self, texts):
        self.texts = texts
        self.emb = EMB_MODEL.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        d = self.emb.shape[1]
        self.index = faiss.IndexFlatIP(d)
        self.index.add(self.emb)

    def search(self, query, k=20):
        q = EMB_MODEL.encode([query], normalize_embeddings=True, convert_to_numpy=True)
        scores, ids = self.index.search(q, k)
        return [(int(i), float(s)) for i in ids[0]]

# ---------------------------
# TOPIC GENERATION
# ---------------------------
def generate_topics(norm_texts):
    topic_model = BERTopic(min_topic_size=15, calculate_probabilities=False, verbose=False)
    topics, _ = topic_model.fit_transform(norm_texts)
    info = topic_model.get_topic_info()
    suggestions = []
    for _, row in info.head(8).iterrows():
        words = [w for w, _ in topic_model.get_topic(row.Topic) or []][:5]
        if words:
            suggestions.append("Exploring: " + ", ".join(words))
    return suggestions

# ---------------------------
# RANKING
# ---------------------------
def recency_weight(pub_date, tau=180):
    try:
        days_old = (datetime.now().date() - datetime.fromisoformat(pub_date).date()).days
        return math.exp(-days_old/tau)
    except:
        return 1.0

def mmr_select(candidates_emb, query_emb, k=5, lambda_=0.7):
    chosen = []
    cand_ids = list(range(len(candidates_emb)))
    sims_to_query = (candidates_emb @ query_emb.T).flatten()
    while cand_ids and len(chosen) < k:
        mmr_scores = []
        for i in cand_ids:
            if not chosen:
                div = 0
            else:
                div = max(candidates_emb[i] @ candidates_emb[j] for j in chosen)
            mmr = lambda_ * sims_to_query[i] - (1-lambda_) * div
            mmr_scores.append((mmr, i))
        mmr_scores.sort(reverse=True)
        _, best = mmr_scores[0]
        chosen.append(best)
        cand_ids.remove(best)
    return chosen

def rank_topics(seed_df, query, level):
    emb = EMB_MODEL.encode(seed_df["norm"].tolist(), normalize_embeddings=True, convert_to_numpy=True)
    q = EMB_MODEL.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0]

    # Select diverse top-5 using MMR
    top_ids = mmr_select(emb, q, k=5)
    ranked = seed_df.iloc[top_ids].copy()

    # Apply recency weights
    ranked["recency"] = ranked["published"].apply(recency_weight)

    # Level adjustment (Beginner prefers simple abstracts)
    if level == "Beginner":
        ranked["score"] = ranked["recency"] * ranked["abstract"].apply(lambda x: 1 if len(x.split())<120 else 0.7)
    elif level == "Intermediate":
        ranked["score"] = ranked["recency"] * 1.0
    else:
        ranked["score"] = ranked["recency"] * ranked["abstract"].apply(lambda x: 1 if len(x.split())>100 else 0.8)

    ranked = ranked.sort_values("score", ascending=False)
    return ranked

# ---------------------------
# FEEDBACK (SQLite)
# ---------------------------
def init_db():
    con = sqlite3.connect("feedback.db")
    con.execute("""CREATE TABLE IF NOT EXISTS feedback(
        ts INTEGER, domain TEXT, level TEXT, title TEXT, vote INTEGER
    )""")
    con.commit()
    con.close()

def save_feedback(domain, level, title, vote):
    con = sqlite3.connect("feedback.db")
    con.execute("INSERT INTO feedback VALUES(?,?,?,?,?)",
                (int(time.time()), domain, level, title, vote))
    con.commit()
    con.close()

init_db()

# ---------------------------
# STREAMLIT UI
# ---------------------------
st.title("🎓 SmartResearch Advisor (Week 4 Prototype)")
st.write("AI-powered research topic generator for students (with ranking + feedback)")

domain = st.selectbox("Choose a domain", ["Artificial Intelligence","Web Development","Renewable Energy"])
level  = st.selectbox("Your level", ["Beginner","Intermediate","Advanced"])

if st.button("Generate Topics"):
    st.info("Fetching and processing data... please wait.")

    # Fetch & preprocess
    df = fetch_arxiv(query=domain, max_results=120)
    df = prepare_corpus(df)

    # Search & generate topics
    idx = VectorIndex(df["norm"].tolist())
    hits = idx.search(f"{domain} {level} research", k=30)
    seed_df = df.iloc[[i for i,_ in hits]]

    topics = generate_topics(seed_df["norm"].tolist())
    ranked = rank_topics(seed_df, f"{domain} {level} research", level)

    st.subheader("✅ Suggested Topics")
    for i, row in ranked.iterrows():
        st.markdown(f"### {row['title']}")
        st.write(row['abstract'][:300] + "...")
        st.markdown(f"[Read Paper]({row['url']})")

        col1, col2 = st.columns(2)
        with col1:
            if st.button(f"👍 {row['title']}", key=f"up{i}"):
                save_feedback(domain, level, row['title'], 1)
                st.success("Feedback recorded")
        with col2:
            if st.button(f"👎 {row['title']}", key=f"down{i}"):
                save_feedback(domain, level, row['title'], -1)
                st.warning("Feedback recorded")

    # Topic modeling suggestions
    st.subheader("🔎 Topic Clusters (from BERTopic)")
    for t in topics:
        st.markdown(f"- {t}")

    # Download option
    st.download_button("📥 Download Papers (CSV)", data=seed_df.to_csv(index=False),
                       file_name="seed_papers.csv", mime="text/csv")

