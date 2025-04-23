import os
import streamlit as st
import openai

# Load Llama API Key
LLAMA_API_KEY = os.getenv("LLAMA_API_KEY")  # Set this in your Hugging Face Space
openai.api_key = LLAMA_API_KEY
openai.api_base = "https://api.llama-api.com/v1"  # Change if your provider gives different base URL

# Function to load text files
def load_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()

# Function to generate a single set of questions
def generate_questions(transcript, co_text, bloom_level):
    prompt = f"""
You are a Question Generator Agent.
Given:
- Full Course Content: {transcript}
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
    response = openai.ChatCompletion.create(
        model="llama-3-70b-instruct",  # Or use whatever model your API supports
        messages=[
            {"role": "system", "content": "You are an expert Question Generator."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=800,
    )
    return response['choices'][0]['message']['content']

# Streamlit App
def main():
    st.title("🎯 Smart Question Generator (Llama API Version)")

    # Load course content and course outcomes
    transcript = load_file("cleaned_transcript.txt")
    course_outcomes = load_file("course_outcomes.txt")
    co_list = course_outcomes.strip().split("\n")  # List of COs
    bloom_levels = ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"]

    # Dropdowns for user selection
    selected_co = st.selectbox("📚 Select Course Outcome:", co_list)
    selected_bloom = st.selectbox("🧠 Select Bloom's Level:", bloom_levels)

    if st.button("🚀 Generate Question"):
        with st.spinner(f"Generating question for '{selected_co}' at '{selected_bloom}' level..."):
            try:
                questions = generate_questions(transcript, selected_co, selected_bloom)
                st.subheader("Generated Questions:")
                st.write(questions)

                # Optional download
                st.download_button("📥 Download Question", questions, file_name="generated_question.txt")

            except Exception as e:
                st.error(f"❗ Error: {e}")

if __name__ == "__main__":
    main()
