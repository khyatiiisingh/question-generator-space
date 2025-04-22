import os
import streamlit as st
import google.generativeai as genai

# Load Gemini API Key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

# Function to load text files
def load_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()

# Function to generate a single set of questions
def generate_questions(content, co_text, bloom_level):
    prompt = f"""
You are a Question Generator Agent.
Given the following:
- Course Content: {content}
- Course Outcome: {co_text}
- Bloom's Taxonomy Level: {bloom_level}
Generate:
- 1 Objective Type Question
- 1 Short Answer Type Question
that map to the given Course Outcome and Bloom's Taxonomy level.
Format:
Objective Question:
1. ...
Short Answer Question:
1. ...
"""
    model = genai.GenerativeModel('gemini-1.5-pro')
    response = model.generate_content(prompt)
    return response.text

# Streamlit UI
def main():
    st.title("Question Generator based on CO + Bloom's Level")

    # Load course content and course outcomes
    transcript = load_file("cleaned_transcript.txt")
    course_outcomes = load_file("course_outcomes.txt")
    co_list = course_outcomes.strip().split("\n")  # List of each CO
    bloom_levels = ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"]

    # Dropdowns for user selection
    selected_co = st.selectbox("📚 Select Course Outcome:", co_list)
    selected_bloom = st.selectbox("🧠 Select Bloom's Level:", bloom_levels)

    if st.button("🚀 Generate Question"):
        with st.spinner(f"Generating question for '{selected_co}' at '{selected_bloom}' level..."):
            questions = generate_questions(transcript, selected_co, selected_bloom)
            st.subheader("Generated Questions:")
            st.write(questions)

if __name__ == "__main__":
    main()
