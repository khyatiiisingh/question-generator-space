import os
import streamlit as st
import google.generativeai as genai

# Load Gemini API Key from Hugging Face Secret Variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

# Function to load text files
def load_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()

# Function to generate questions
def generate_questions(content, co_text, bloom_level):
    prompt = f"""
You are a Question Generator Agent.

Given the following:
- Course Content: {content}
- Course Outcome: {co_text}
- Bloom's Taxonomy Level: {bloom_level}

Generate:
- 2 Objective Type Questions
- 2 Short Answer Type Questions
that map to the given Course Outcome and Bloom's Taxonomy level.

Format:
Objective Questions:
1. ...
2. ...

Short Answer Questions:
1. ...
2. ...
"""
    model = genai.GenerativeModel('gemini-1.5-pro')  # Correct Gemini model
    response = model.generate_content(prompt)
    return response.text

# Streamlit UI
def main():
    st.title("🌟 Course Outcome + Bloom's Taxonomy Question Generator")

    # Load the fixed transcript and course outcome files
    transcript = load_file("cleaned_transcript.txt")
    course_outcomes = load_file("course_outcomes.txt")

    if st.button("🚀 Generate Questions"):
        all_questions = ""
        co_list = course_outcomes.strip().split("\n")  # List of each CO
        bloom_levels = ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"]

        for co in co_list:
            st.subheader(f"📚 {co}")
            all_questions += f"\n\n## {co}\n\n"
            for bloom in bloom_levels:
                st.markdown(f"### 🌟 Bloom Level: {bloom}")
                with st.spinner(f"Generating questions for {co} at {bloom} level..."):
                    questions = generate_questions(transcript, co, bloom)
                    st.write(questions)
                    all_questions += f"\n\n### {bloom}\n{questions}\n"

        # Option to download all generated questions
        st.download_button("📥 Download All Questions", all_questions, file_name="generated_questions.txt")

if __name__ == "__main__":
    main()
