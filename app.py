import os
import streamlit as st
import google.generativeai as genai

# Load Gemini API Key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

# Function to load text file
def load_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()

# Function to split large text into chunks
def split_text(text, max_words=500):
    words = text.split()
    chunks = [' '.join(words[i:i+max_words]) for i in range(0, len(words), max_words)]
    return chunks

# Function to summarize each chunk
def summarize_chunk(chunk):
    summarization_prompt = f"""
Summarize the following course content chunk in 50-70 words:

Chunk:
{chunk}

Summary:
"""
    model = genai.GenerativeModel('gemini-1.5-pro')
    response = model.generate_content(summarization_prompt)
    return response.text

# Function to create full course summary
def create_summary(full_text):
    chunks = split_text(full_text, max_words=500)
    full_summary = ""

    model = genai.GenerativeModel('gemini-1.5-pro')

    for i, chunk in enumerate(chunks):
        with st.spinner(f"Summarizing chunk {i+1}/{len(chunks)}..."):
            summary = summarize_chunk(chunk)
            full_summary += summary + " "
    
    return full_summary

# Function to generate question
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
    st.title("🎯 Smart Question Generator (Chunk-based Summarization)")

    # Load course content and COs
    transcript = load_file("cleaned_transcript.txt")
    course_outcomes = load_file("course_outcomes.txt")
    co_list = course_outcomes.strip().split("\n")
    bloom_levels = ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"]

    # Summarize course content (smaller chunks)
    if st.button("🔎 Summarize Course Content"):
        course_summary = create_summary(transcript)
        st.session_state['course_summary'] = course_summary
        st.success("✅ Course summarized successfully!")

    if 'course_summary' in st.session_state:
        selected_co = st.selectbox("📚 Select Course Outcome:", co_list)
        selected_bloom = st.selectbox("🧠 Select Bloom's Level:", bloom_levels)

        if st.button("🚀 Generate Question"):
            with st.spinner(f"Generating question for '{selected_co}' at '{selected_bloom}' level..."):
                try:
                    questions = generate_questions(st.session_state['course_summary'], selected_co, selected_bloom)
                    st.subheader("Generated Questions:")
                    st.write(questions)
                    st.download_button("📥 Download Question", questions, file_name="generated_question.txt")
                except Exception as e:
                    st.error(f"❗ Error: {e}")

if __name__ == "__main__":
    main()
