from google.generativeai import configure, GenerativeModel
import os

os.environ["GOOGLE_API_KEY"] = "AIzaSyYourBossKeyHere"
configure(api_key=os.environ["GOOGLE_API_KEY"])

model = GenerativeModel("gemini-1.5-pro")
response = model.generate_content("hi, are you working?")
print(response.text)
