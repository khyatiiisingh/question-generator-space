# === FILE: app.py (Cleaned – No Test Mode / No Quota Logic) ===

import os
import time
import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
import pandas as pd
from datetime import datetime
import re
import warnings
import logging

# === Suppress Warnings ===
warnings.filterwarnings("ignore", category=UserWarning, module="langchain")
logging.getLogger("streamlit.runtime.scriptrunner.script_runner").setLevel(logging.ERROR)
logging.getLogger("streamlit.watcher.local_sources_watcher").setLevel(logging.ERROR)

# === Load API key ===
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# === Constants ===
DATA_PATH = "cleaned_transcript.txt"
CO_PATH = "course_outcomes.txt"
CACHE_PATH = "question_cache.csv"

# === Static CO → PO Mapping ===
CO_PO_MAP = {
    "1. identify different types of concrete and its properties": ["PO1", "PO2"],
    "2. determine the workability of concrete": ["PO2", "PO4"],
    "3. determine strength and durability of concrete": ["PO3", "PO5"],
    "4. design concrete mixes for the given conditions": ["PO3", "PO6"],
    "5. perform tests of hardened concrete": ["PO4", "PO6"],
    "6. select types of admixture and special concrete for given condition.": ["PO5", "PO7"]
}

# === Vector DB Setup + Topic Extractor ===
@st.cache_resource(show_spinner=False)
def load_vector_db():
    loader = TextLoader(DATA_PATH)
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = text_splitter.split_documents(data)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = FAISS.from_documents(chunks, embeddings)
    return vectordb, chunks

vectordb, chunks = load_vector_db()

@st.cache_data(show_spinner=False)
def extract_topics(chunks):
    topics = set()
    for chunk in chunks:
        lines = chunk.page_content.split("\n")
        for line in lines:
            line = line.strip()
            if len(line) > 10 and not line.lower().startswith("course") and not line.lower().startswith("unit"):
                if re.match(r'^[A-Z].*', line):
                    topics.add(line)
    return sorted(topics)

extracted_topics = extract_topics(chunks)

# === Gemini Setup ===
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.7)

def get_response(prompt):
    return llm.invoke(prompt)

# === Prompt Template ===
prompt_template = PromptTemplate(
    input_variables=["topic", "blooms", "co", "po", "qtype", "marks", "assessment"],
    template="""
Generate a {qtype} question from topic: {topic}
Targeting:
- Bloom Level: {blooms}
- Course Outcome: {co}
- Program Outcomes: {po}
- Marks: {marks}
- Assessment Type: {assessment}
Only provide the question.
"""
)

# === Cache Handling ===
def load_cache():
    if os.path.exists(CACHE_PATH):
        return pd.read_csv(CACHE_PATH)
    else:
        return pd.DataFrame(columns=["topic", "blooms", "co", "po", "qtype", "marks", "assessment", "question"])

def save_to_cache(df):
    df.to_csv(CACHE_PATH, index=False)

def check_cache(df, topic, blooms, co, po, qtype, marks, assessment):
    filtered = df[(df.topic == topic) & (df.blooms == blooms) & (df.co == co) & (df.po == po) &
                  (df.qtype == qtype) & (df.marks == str(marks)) & (df.assessment == assessment)]
    if not filtered.empty:
        return filtered.iloc[0].question
    return None

# === Streamlit UI ===
st.title("📘 Gemini-Pro CO-PO Based Question Generator")

# --- Load COs ---
with open(CO_PATH) as f:
    course_outcomes = [line.strip() for line in f if line.strip() and line[0].isdigit()]

st.markdown("### Step 1: Select Topic")
topic = st.selectbox("Select a topic:", options=extracted_topics)

st.markdown("### Step 2: Select Course Outcome")
co_selected = st.selectbox("Select Course Outcome:", course_outcomes)
po_list = CO_PO_MAP.get(co_selected.strip(), [])
po_display = ", ".join(po_list) if po_list else "Not mapped"
st.markdown(f"**Mapped Program Outcomes:** {po_display}")

st.markdown("### Step 3: Select Bloom Level + Question Type + Marks")
bloom_level = st.selectbox("Select Bloom's Level:", ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"])
qtype = st.selectbox("Select Question Type:", ["MCQ", "Short Answer", "Long Answer"])
marks = st.selectbox("Marks: ", [1, 2, 5, 10])

st.markdown("### Step 4: Select Assessment Type")
assessment = st.selectbox("Assessment Type:", ["IA1", "IA2", "Midterm", "Endsem", "Research"])

if st.button("Generate Question"):
    if topic:
        with st.spinner("Generating question..."):
            cache_df = load_cache()
            cached_question = check_cache(cache_df, topic, bloom_level, co_selected, po_display, qtype, str(marks), assessment)

            if cached_question:
                st.info("💾 Loaded from cache")
                st.text_area("Generated Question:", value=cached_question, height=150)
            else:
                prompt = prompt_template.format(
                    topic=topic,
                    blooms=bloom_level,
                    co=co_selected,
                    po=po_display,
                    qtype=qtype,
                    marks=marks,
                    assessment=assessment
                )
                try:
                    response = get_response(prompt)
                    question = response.content.strip()
                    st.text_area("Generated Question:", value=question, height=150)
                    new_row = pd.DataFrame([[topic, bloom_level, co_selected, po_display, qtype, marks, assessment, question]],
                                           columns=["topic", "blooms", "co", "po", "qtype", "marks", "assessment", "question"])
                    cache_df = pd.concat([cache_df, new_row], ignore_index=True)
                    save_to_cache(cache_df)
                except Exception as e:
                    st.error(f"❌ Gemini error: {e}")
    else:
        st.warning("Please select a topic.")
