import os
import ssl
import certifi
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

ssl._create_default_https_context = ssl._create_unverified_context
os.environ["PYTHONHTTPSVERIFY"] = "0"
os.environ["SSL_CERT_FILE"] = certifi.where()

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")    

if not api_key:
    raise ValueError("Missing GOOGLE_API_KEY in .env file!")

llm = ChatGoogleGenerativeAI(
    model = "gemini-3.6-flash", google_api_key = api_key, temperature=0
    )

#smoke test
print("Calling Gemini...")
res = llm.invoke("Say 'ready'")
print("Gemini response:", res.content)