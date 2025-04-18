import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Initialize Gemini LLM
llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_API_KEY)

# Pre-uploaded cleaned transcript
with open("cleaned_transcript.txt", "r", encoding="utf-8") as file:
    transcript = file.read()

# Predefined Course Outcomes
course_outcomes = {
    "CO1": "Understand the properties of fresh and hardened concrete.",
    "CO2": "Analyze the behavior of concrete under different loading conditions.",
    "CO3": "Design concrete mixes based on IS codes.",
    "CO4": "Evaluate durability and sustainability of concrete structures.",
}

# Bloom's Taxonomy Levels
blooms_levels = [
    "Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"
]

# Streamlit UI
st.set_page_config(page_title="LMS Question Generator", layout="wide")
st.title("🎯 LMS Question Generator (Auto-mapped with COs & Bloom's Taxonomy)")

if st.button("🚀 Generate Questions"):
    with st.spinner("Generating questions... please wait..."):
        for co_key, co_description in course_outcomes.items():
            st.header(f"📘 Course Outcome: {co_key} - {co_description}")
            
            for bloom_level in blooms_levels:
                prompt = f"""
You are an expert question generator for a university LMS system.

Given the following:

- **Course Outcome**: {co_key} - {co_description}
- **Bloom's Taxonomy Level**: {bloom_level}
- **Course Content**: {transcript}

Generate:
1. 🔹 2 Objective Type Questions (MCQ or Fill in the blanks)
2. 🔹 2 Short Answer Type Questions

Ensure each question is aligned with the specified Course Outcome and Bloom's level.
Show output in a clean numbered list.
"""
                response = llm.invoke(prompt)
                st.subheader(f"🧠 Bloom's Level: {bloom_level}")
                st.markdown(response.content)
