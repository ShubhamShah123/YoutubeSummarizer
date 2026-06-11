import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
print("[+] Setup functions libraries imported!")

load_dotenv()
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

# --- Singletons: created once per session, reused across calls ---
_llm_instance = None
_embedding_instance = None


def setup_credentials():
	print("[+] Setting up credentials.")
	model_id = "gemini-2.5-flash-lite"
	return model_id


def define_parameters():
	print("[+] Defining Parameters.")
	return {
		'temperature': 0.5,
		'max_tokens': 900
	}


def setup_model(model_id):
	global _llm_instance
	if _llm_instance is not None:
		print(f"[+] Reusing cached LLM: {model_id}")
		return _llm_instance
	print(f"[+] Init Model: {model_id}")
	params = define_parameters()
	_llm_instance = ChatGoogleGenerativeAI(
		model=model_id,
		temperature=params.get('temperature'),
		google_api_key=GEMINI_API_KEY,
		max_output_tokens=params.get('max_tokens')
	)
	return _llm_instance


def setup_embedding():
	global _embedding_instance
	if _embedding_instance is not None:
		print("[+] Reusing cached embedding model.")
		return _embedding_instance
	print("[+] Setting up embedding.")
	_embedding_instance = HuggingFaceEmbeddings(
		model_name="sentence-transformers/all-MiniLM-L6-v2"
	)
	return _embedding_instance