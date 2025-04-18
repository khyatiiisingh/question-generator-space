import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os

# Load API key from .env
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Load Gemini LLM
llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_API_KEY)

# Streamlit UI
st.set_page_config(page_title="LMS Question Generator", layout="wide")
st.title("🎓 LMS Question Generator using Bloom's Taxonomy")

# Upload section
transcript_file = st.file_uploader("📄 Upload Course Transcript (.txt)", type=["txt"])
course_outcomes = st.text_area("📘 Enter Course Outcomes (COs)")

if transcript_file:
    transcript = transcript_file.read().decode("utf-8")
else:
    transcript = ""

# Prompt template
template = """
You are an expert question generator for a university LMS system.

Based on the provided course transcript and course outcomes (COs), generate:

1. 🔹 3 Objective Questions (MCQs or Fill in the blanks)
2. 🔹 3 Short Answer Questions

Each question should:
- Clearly mention the corresponding Course Outcome (e.g., CO1, CO2, etc.)
- Specify its Bloom's Taxonomy Level (e.g., Remember, Understand, Apply, Analyze, Evaluate, Create)

### Course Outcomes:
{course_outcomes}

### Course Transcript:
{transcript}

Now generate the questions.
"""

prompt = PromptTemplate.from_template(template)

if st.button("🚀 Generate Questions"):
    if transcript and course_outcomes:
        with st.spinner("Generating questions..."):
            full_prompt = prompt.format(course_outcomes=course_outcomes, transcript=transcript)
            response = llm.invoke(full_prompt)
            st.markdown("### Generated Questions:")
            st.markdown(response.content)
    else:
        st.warning("Please upload a transcript and enter course outcomes.")
