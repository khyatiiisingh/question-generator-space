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

# Load text files
def load_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()

# Chunking logic
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

# Save and Load vector index
def save_vector_data(index, chunks, embeddings):
    faiss.write_index(index, "faiss_index.index")
    np.save("chunks.npy", np.array(chunks))
    np.save("embeddings.npy", embeddings)

def load_vector_data():
    if os.path.exists("faiss_index.index") and os.path.exists("chunks.npy") and os.path.exists("embeddings.npy"):
        index = faiss.read_index("faiss_index.index")
        chunks = np.load("chunks.npy", allow_pickle=True).tolist()
        embeddings = np.load("embeddings.npy")
        return index, chunks, embeddings
    return None, None, None

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

# Chain of Thought + Clean Question Output
def generate_questions(retrieved_content, co_text, bloom_level):
    prompt_parts = [
        "You are a Question Generator Agent.",
        f"Course Outcome (CO): {co_text}",
        f"Bloom's Taxonomy Level: {bloom_level}",
        "Based on the content below, generate two questions:",
        "- One Objective Type",
        "- One Short Answer Type",
        "Content:\n" + retrieved_content,
        "\nOnly output the questions in the following format:",
        "Objective Question:\n1. <question>",
        "Short Answer Question:\n1. <question>"
    ]
    full_prompt = "\n".join(prompt_parts)
    model = genai.GenerativeModel('gemini-1.5-pro')
    response = model.generate_content(full_prompt)

    output = response.text.strip()
    if "Objective Question" in output:
        output = output.split("Objective Question", 1)[1]
        output = "Objective Question" + output.strip()
    return output

# Streamlit App
def main():
    st.title("CO & Bloom Level Based Question Generator (with Handout Support)")

    # Load base files
    transcript = load_file("cleaned_transcript.txt")
    course_outcomes = load_file("course_outcomes.txt")
    co_list = course_outcomes.strip().split("\n")
    bloom_levels = ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"]

    # Upload course handout file
    handout_file = st.file_uploader("Upload Course Handout (txt)", type=["txt"])
    handout_content = ""
    if handout_file is not None:
        handout_content = handout_file.read().decode("utf-8")

    # Combine handout + transcript
    combined_text = transcript + "\n" + handout_content if handout_content else transcript

    # Load or build vector DB
    index, chunks, embeddings = load_vector_data()
    if index is None:
        st.info("Building vector database... please wait")
        chunks = chunk_text(combined_text)
        index, chunks, embeddings = build_vector_index(chunks)
        save_vector_data(index, chunks, embeddings)
        st.success("Vector database built and cached")
    else:
        st.success("Loaded cached vector database")

    # Select CO and Bloom Level
    selected_co = st.selectbox("Select Course Outcome:", co_list)
    selected_bloom = st.selectbox("Select Bloom's Level:", bloom_levels)

    if st.button("Generate Question"):
        with st.spinner("Retrieving content and generating question..."):
            try:
                best_chunk = semantic_search(selected_co, index, chunks, embeddings, top_k=1)[0]
                questions = generate_questions(best_chunk, selected_co, selected_bloom)
                st.subheader("Generated Questions")
                st.write(questions)
                st.download_button("Download Question", questions, file_name="generated_question.txt")
            except Exception as e:
                st.error(f"Error: {e}")

if __name__ == "__main__":
    main()
