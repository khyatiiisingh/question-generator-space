import os
import streamlit as st
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# Load Gemini API Key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

# Load sentence transformer model
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

# Load Files
def load_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()

# Chunking the transcript
def chunk_text(text, chunk_size=500):
    sentences = text.split('. ')
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= chunk_size:
            current_chunk += sentence + ". "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "

    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

# Build Vector Index
def build_vector_index(chunks):
    embeddings = embed_model.encode(chunks)
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))

    return index, chunks, embeddings

# Semantic search
def semantic_search(co_text, index, chunks, embeddings, top_k=1):
    co_embedding = embed_model.encode([co_text])
    distances, indices = index.search(np.array(co_embedding), top_k)
    retrieved_chunks = [chunks[i] for i in indices[0]]
    return retrieved_chunks

# Question Generation
def generate_questions(retrieved_content, co_text, bloom_level):
    prompt = f"""
You are a Question Generator Agent.

Given:
- Content Chunk: {retrieved_content}
- Course Outcome (CO): {co_text}
- Bloom's Taxonomy Level: {bloom_level}

Generate:
- 1 Objective Type Question
- 1 Short Answer Type Question
based only on the given content.

Format:
Objective Question:
1. ...

Short Answer Question:
1. ...
"""
    model = genai.GenerativeModel('gemini-1.5-pro')
    response = model.generate_content(prompt)
    return response.text

# Streamlit App
def main():
    st.title(" CO & Bloom Level Based Question Generator")

    # Load course content and course outcomes
    transcript = load_file("cleaned_transcript.txt")
    course_outcomes = load_file("course_outcomes.txt")
    co_list = course_outcomes.strip().split("\n")
    bloom_levels = ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"]

    # Chunking and Building Index
    st.info("Building vector database... please wait ⏳")
    chunks = chunk_text(transcript)
    index, chunks, embeddings = build_vector_index(chunks)
    st.success("✅ Vector database built successfully!")

    # Select CO and Bloom Level
    selected_co = st.selectbox("📚 Select Course Outcome:", co_list)
    selected_bloom = st.selectbox("🧠 Select Bloom's Level:", bloom_levels)

    if st.button("🚀 Generate Question"):
        with st.spinner("Fetching best matching content and generating question..."):
            try:
                # Semantic Search
                best_chunk = semantic_search(selected_co, index, chunks, embeddings, top_k=1)[0]

                # Generate Questions
                questions = generate_questions(best_chunk, selected_co, selected_bloom)

                st.subheader("Generated Questions:")
                st.write(questions)

                # Optional download
                st.download_button("📥 Download Question", questions, file_name="generated_question.txt")
            except Exception as e:
                st.error(f"❗ Error: {e}")

if __name__ == "__main__":
    main()
