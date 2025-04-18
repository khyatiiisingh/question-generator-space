import os
import streamlit as st
import google.generativeai as genai

# Load Gemini API Key from Hugging Face Secret Variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

# Function to load text file content
def load_file_from_streamlit(uploaded_file):
    return uploaded_file.read().decode("utf-8")

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
    model = genai.GenerativeModel('gemini-1.5-pro')  # Correct model
    response = model.generate_content(prompt)
    return response.text

# Streamlit UI
def main():
    st.title("🌟 Course Outcome + Bloom's Taxonomy Question Generator")

    # Upload files
    uploaded_transcript = st.file_uploader("📄 Upload the Course Transcript (.txt)", type=["txt"])
    uploaded_course_outcome = st.file_uploader("📄 Upload the Course Outcomes (.txt)", type=["txt"])

    if uploaded_transcript and uploaded_course_outcome:
        transcript = load_file_from_streamlit(uploaded_transcript)
        course_outcomes = load_file_from_streamlit(uploaded_course_outcome)

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

            # Download all generated questions
            st.download_button("📥 Download All Questions", all_questions, file_name="generated_questions.txt")

    else:
        st.warning("⚠️ Please upload both the Course Transcript and Course Outcomes files!")

if __name__ == "__main__":
    main()
