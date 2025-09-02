import streamlit as st
import pandas as pd
import arxiv
from sentence_transformers import SentenceTransformer, util
import spacy
import en_core_web_sm

# Load NLP tools
nlp = en_core_web_sm.load()
embedder = SentenceTransformer("all-MiniLM-L6-v2")

st.title("🎓 SmartResearch Advisor (Light Version)")
st.write("Fast & lightweight topic generator for Streamlit Cloud")

# User input
domain = st.text_input("Enter your research domain (e.g., Machine Learning, Healthcare)")
level = st.selectbox("Select your level", ["Undergraduate", "Postgraduate", "PhD"])

if st.button("Generate Topics"):
    if not domain:
        st.warning("Please enter a domain.")
    else:
        with st.spinner("Fetching research papers..."):
            # Fetch papers from arXiv
            search = arxiv.Search(
                query=domain,
                max_results=20,
                sort_by=arxiv.SortCriterion.SubmittedDate
            )
            papers = []
            for paper in search.results():
                papers.append({"title": paper.title, "summary": paper.summary})
            
            df = pd.DataFrame(papers)
            
            if not df.empty:
                st.success("Found some papers! Generating topics...")
                
                # Embed paper titles
                embeddings = embedder.encode(df["title"].tolist(), convert_to_tensor=True)
                
                # Pick top 5 diverse titles as "topics"
                unique_titles = list(set(df["title"].tolist()))[:5]
                st.subheader("🔑 Suggested Research Topics")
                for i, title in enumerate(unique_titles, 1):
                    st.write(f"**{i}. {title}**")
            else:
                st.error("No papers found. Try another domain.")
