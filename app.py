import time
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted

# Configure Gemini API Key
genai.configure(api_key="YOUR_GEMINI_API_KEY")

# Load the Gemini model
model = genai.GenerativeModel("gemini-1.5-pro")

# Function to create a prompt
def create_prompt(transcript, course_outcomes, bloom_level):
    return f"""You are a smart question-generating agent.
Based on the given course outcomes and Bloom's Taxonomy cognitive level, generate 5 questions related to the provided transcript content.

Transcript:
\"\"\"
{transcript}
\"\"\"

Course Outcomes:
{course_outcomes}

Bloom's Level:
{bloom_level}

Guidelines:
- The questions should strictly match the Bloom's level cognitive action.
- The questions must be relevant to the transcript content.
- Prefer clear, focused, and diverse question styles.

Format:
1. [Question 1]
2. [Question 2]
...
"""

# Retry wrapper
def generate_questions(transcript, course_outcomes, bloom_level, retries=3, delay=20):
    prompt = create_prompt(transcript, course_outcomes, bloom_level)

    for attempt in range(retries):
        try:
            response = model.generate_content(prompt)
            return response.text
        except ResourceExhausted:
            print(f"[!] Quota exceeded. Attempt {attempt+1} of {retries}. Waiting {delay} seconds...")
            time.sleep(delay)
        except Exception as e:
            print(f"[!] Unexpected error: {e}")
            break

    return "[x] Failed to generate questions after retries."

# Main function
def main():
    # Load transcript
    with open("cleaned_transcript.txt", "r") as f:
        transcript = f.read()

    # Your 5 course outcomes
    course_outcomes = """
    1. Understand the Fundamental Properties of Concrete and Its Components.
    2. Describe the Manufacturing Process and Environmental Impact of Cement.
    3. Analyze the Chemical Composition and Hydration Reactions of Cement Compounds.
    4. Interpret the Microstructure and Physical Changes During Concrete Setting and Hardening.
    5. Evaluate the Properties and Classification of Modern Engineered Concrete.
    """

    # All Bloom's Taxonomy levels
    blooms_levels = [
        "Remember",
        "Understand",
        "Apply",
        "Analyze",
        "Evaluate",
        "Create"
    ]

    all_questions = {}

    for bloom_level in blooms_levels:
        print(f"\n=== Generating Questions for Bloom's Level: {bloom_level} ===")
        questions = generate_questions(transcript, course_outcomes, bloom_level)
        all_questions[bloom_level] = questions
        print(questions)
        time.sleep(5)  # slight wait to respect rate limits better

    # Save to file
    with open("generated_questions.txt", "w") as f:
        for bloom_level, questions in all_questions.items():
            f.write(f"\n=== {bloom_level} ===\n")
            f.write(questions + "\n")

    print("\n✅ All questions saved to 'generated_questions.txt'.")

if __name__ == "__main__":
    main()
