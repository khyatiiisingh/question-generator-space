import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Configure Gemini API
genai.configure(api_key=GOOGLE_API_KEY)

# Load cleaned transcript
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

# Streamlit App
st.set_page_config(page_title="LMS Question Generator", layout="wide")
st.title("🎯 LMS Question Generator (CO + Bloom's Taxonomy)")

if st.button("🚀 Generate Questions"):
    with st.spinner("Generating questions... please wait..."):
        model = genai.GenerativeModel('gemini-pro')  # using correct Gemini model
        
        for co_key, co_description in course_outcomes.items():
            st.header(f"📘 Course Outcome: {co_key} - {co_description}")
            
            for bloom_level in blooms_levels:
                prompt = f"""
You are a professional exam question designer.

Given:

- **Course Outcome**: {co_key} - {co_description}
- **Bloom's Taxonomy Level**: {bloom_level}
- **Course Content**: {transcript}

Please generate:
1. 🔹 2 Objective Type Questions (MCQ or Fill in the blanks)
2. 🔹 2 Short Answer Type Questions

Ensure that questions align strictly with the given Course Outcome and Bloom's level.
Show the output in clean numbered format.
"""

                response = model.generate_content(prompt)
                generated_text = response.text

                st.subheader(f"🧠 Bloom's Level: {bloom_level}")
                st.markdown(generated_text)
