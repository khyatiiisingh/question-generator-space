import streamlit as st
from langchain.llms import OpenAI
import os

# Load your OpenAI key from environment variables (you can set it in Huggingface secrets)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize LLM
llm = OpenAI(temperature=0.5, openai_api_key=OPENAI_API_KEY)

# Load Transcript
def load_transcript(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

# Load Course Outcomes
def load_course_outcomes(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

# Generate Questions
def generate_questions(transcript, course_outcomes, bloom_level, question_type):
    prompt = f"""
You are a Course Content Expert.

You have:
- Course Content: {transcript}
- Course Outcomes: {course_outcomes}

Task:
- Generate 5 {question_type} questions.
- Follow Bloom's Taxonomy level: {bloom_level}.
- Align questions properly with the given Course Outcomes.
- Present questions clearly and separate each by new line.

Important:
- If {question_type} is Objective, generate MCQ type questions with 4 options (a-d) and mention correct answer.
- If {question_type} is Short Answer, keep questions concise and direct.

Start Generating:
"""
    response = llm.invoke(prompt)
    return response

# Streamlit App
def main():
    st.set_page_config(page_title="LMS Question Generator", page_icon="📚")
    st.title("📚 LMS Agent: Question Generator")
    st.write("Generate Course Outcome Based Questions Following Bloom's Taxonomy!")

    transcript = load_transcript("cleaned_transcript.txt")
    course_outcomes = load_course_outcomes("course_outcomes.txt")

    # Select Bloom's Level
    bloom_levels = ["Remembering", "Understanding", "Applying", "Analyzing", "Evaluating", "Creating"]
    selected_level = st.selectbox("Select Bloom's Taxonomy Level:", bloom_levels)

    # Select Question Type
    question_types = ["Objective (MCQ)", "Short Answer"]
    selected_qtype = st.selectbox("Select Question Type:", question_types)

    if st.button("🚀 Generate Questions"):
        with st.spinner("Generating Questions..."):
            questions = generate_questions(
                transcript,
                course_outcomes,
                bloom_level=selected_level,
                question_type=selected_qtype
            )
            st.success("Questions Generated Successfully!")
            st.write(questions)

if __name__ == "__main__":
    main()
