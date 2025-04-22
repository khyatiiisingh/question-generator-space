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

# Function to summarize course content
def summarize_content(content):
    summarization_prompt = f"""
Summarize the following course content into 100-150 words, keeping the important concepts only:

Content:
{content}

Summary:
"""
    model = genai.GenerativeModel('gemini-1.5-pro')
    response = model.generate_content(summarization_prompt)
    return response.text

# Function to generate a single set of questions
def generate_questions(summary, co_text, bloom_level):
    prompt = f"""
You are a Question Generator Agent.

Given:
- Summary of Course Content: {summary}
- Course Outcome (CO): {co_text}
- Bloom's Taxonomy Level: {bloom_level}

Generate:
- 1 Objective Type Question
- 1 Short Answer Type Question
that map properly to the given CO and Bloom's Level.

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
    st.title("🎯 Smart Question Generator (Summary Based)")

    # Load course content and course outcomes
    transcript = load_file("cleaned_transcript.txt")
    course_outcomes = load_file("course_outcomes.txt")
    co_list = course_outcomes.strip().split("\n")  # List of COs
    bloom_levels = ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"]

    # Summarize course content once
    with st.spinner("🔎 Summarizing course content..."):
        course_summary = summarize_content(transcript)
    st.success("✅ Course content summarized successfully!")

    # Dropdowns for user selection
    selected_co = st.selectbox("📚 Select Course Outcome:", co_list)
    selected_bloom = st.selectbox("🧠 Select Bloom's Level:", bloom_levels)

    if st.button("🚀 Generate Question"):
        with st.spinner(f"Generating question for '{selected_co}' at '{selected_bloom}' level..."):
            try:
                questions = generate_questions(course_summary, selected_co, selected_bloom)
                st.subheader("Generated Questions:")
                st.write(questions)

                # Optional download
                st.download_button("📥 Download Question", questions, file_name="generated_question.txt")

            except Exception as e:
                st.error(f"❗ Error: {e}")

if __name__ == "__main__":
    main()
