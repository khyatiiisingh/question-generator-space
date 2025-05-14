# === Dhamm AI: CO-PO Mapped Question Generator ===

import os
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from tenacity import retry, wait_random_exponential, stop_after_attempt
import pandas as pd
import uuid

# === Load environment variables ===
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# === Initialize Gemini LLM ===
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.7)

@retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(3))
def generate_question(prompt):
    return llm.invoke(prompt)

# === Load Course Outcomes and CO-PO Mapping ===
def load_course_outcomes():
    with open("course_outcomes.txt") as f:
        return [line.strip() for line in f.readlines() if line.strip() and not line.startswith("Course Outcomes")]

def load_copo_mapping():
    return {
        "CO1": ["PO1", "PO2"],
        "CO2": ["PO3"],
        "CO3": ["PO2", "PO5"],
        "CO4": ["PO4", "PO6"],
        "CO5": ["PO3", "PO7"],
        "CO6": ["PO8"]
    }

# === Save Question to Local Bank ===
def save_to_question_bank(data, file_path="question_bank.csv"):
    df = pd.DataFrame([data])
    if os.path.exists(file_path):
        df.to_csv(file_path, mode='a', header=False, index=False)
    else:
        df.to_csv(file_path, index=False)

# === Streamlit UI ===
st.set_page_config(page_title="CO-PO Based Question Generator", layout="centered")
st.title("📘 Based Question Generator")

course_outcomes = load_course_outcomes()
copo_map = load_copo_mapping()
blooms_levels = ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"]
question_types = ["MCQ", "Short Answer", "Long Answer"]
assessment_types = ["IA1", "IA2", "Mid Term", "End Term", "Research Assessment"]

st.success("Loaded cached vector database")

selected_co = st.selectbox("Select Course Outcome:", options=course_outcomes)
selected_bloom = st.selectbox("Select Bloom's Level:", options=blooms_levels)
selected_type = st.selectbox("Select Question Type:", options=question_types)
selected_marks = st.slider("Select Marks:", 1, 20, 5)
selected_assessment = st.selectbox("Select Assessment Type:", options=assessment_types)
topic = st.text_input("Enter Topic Name:")

if st.button("Generate Question"):
    if not topic:
        st.warning("Please enter a topic.")
    else:
        po_list = copo_map.get(f"CO{course_outcomes.index(selected_co)+1}", [])
        prompt = f"""
        Generate a {selected_type} question from topic "{topic}".
        - Bloom's Taxonomy Level: {selected_bloom}
        - Target Course Outcome: {selected_co}
        - Target Program Outcomes: {', '.join(po_list)}
        - Marks: {selected_marks}
        - Assessment Type: {selected_assessment}
        Provide only the question text.
        """
        try:
            question = generate_question(prompt)
            st.success("Question Generated Successfully!")
            st.markdown(f"**Q:** {question}")

            save_to_question_bank({
                "id": str(uuid.uuid4()),
                "topic": topic,
                "question": question,
                "type": selected_type,
                "bloom_level": selected_bloom,
                "co": selected_co,
                "po": ', '.join(po_list),
                "marks": selected_marks,
                "assessment_type": selected_assessment
            })

        except Exception as e:
            st.error(f"❌ Error: {e}")
