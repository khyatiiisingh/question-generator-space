# === FILE: app.py ===

import os
import time
import streamlit as st
from dotenv import load_dotenv
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from tenacity import retry, wait_random_exponential, stop_after_attempt

# === Load API key ===
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# === Constants ===
DATA_PATH = "cleaned_transcript.txt"
CO_PATH = "course_outcomes.txt"

# === Vector DB Setup ===
@st.cache_resource(show_spinner=False)
def load_vector_db():
    loader = TextLoader(DATA_PATH)
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(data)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = FAISS.from_documents(chunks, embeddings)
    return vectordb

vectordb = load_vector_db()
st.success("✅ Loaded cached vector database")

# === LLM Setup ===
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.7)

@retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(3))
def get_response(prompt):
    return llm.invoke(prompt)

# === Prompt Template ===
prompt_template = PromptTemplate(
    input_variables=["context", "topic", "blooms", "co", "qtype", "marks", "assessment"],
    template="""
Based on the following course content:
{context}

Generate a {qtype} question from the topic '{topic}' that matches:
- Bloom's Taxonomy level: {blooms}
- Course Outcome: {co}
- Marks: {marks}
- Assessment Type: {assessment} (e.g., IA1, IA2, Midterm, Endsem)

Question:
"""
)

# === Streamlit UI ===
st.title("📘 Outcome-Based Question Generator")

# --- Load COs ---
with open(CO_PATH) as f:
    course_outcomes = [line.strip() for line in f if line.strip() and line[0].isdigit()]

topic = st.text_input("Enter Topic:")
co_selected = st.selectbox("Select Course Outcome:", course_outcomes)
bloom_level = st.selectbox("Select Bloom's Level:", ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"])
qtype = st.selectbox("Select Question Type:", ["MCQ", "Short Answer", "Long Answer"])
marks = st.selectbox("Marks: ", [1, 2, 5, 10])
assessment = st.selectbox("Assessment Type:", ["IA1", "IA2", "Midterm", "Endsem", "Research"])

if st.button("Generate Question"):
    if topic:
        with st.spinner("Generating question..."):
            context_docs = vectordb.similarity_search(topic, k=3)
            context = "\n\n".join([doc.page_content for doc in context_docs])
            prompt = prompt_template.format(
                context=context,
                topic=topic,
                blooms=bloom_level,
                co=co_selected,
                qtype=qtype,
                marks=marks,
                assessment=assessment
            )
            try:
                response = get_response(prompt)
                st.text_area("Generated Question:", value=response.content.strip(), height=150)
            except Exception as e:
                st.error(f"❌ Failed to generate question: {e}")
    else:
        st.warning("Please enter a topic.")
