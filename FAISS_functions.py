import os
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
print("[+] FAISS Functions libraries imported!")

processed_transcript = ""

CACHE_DIR = "./cache"



def get_faiss_index_path(video_id):
	return os.path.join(CACHE_DIR, f"{video_id}_faiss_index")


def create_faiss_index(chunks, embedding_model, video_id=None):
	"""Creates a FAISS index from chunks. Saves to disk if video_id is given.
	   Loads from disk on subsequent calls for the same video."""
	if video_id:
		index_path = get_faiss_index_path(video_id)
		if os.path.exists(index_path):
			print(f"[+] Loading FAISS index from cache: {index_path}")
			return FAISS.load_local(index_path, embedding_model, allow_dangerous_deserialization=True)

	print("[+] Creating FAISS Index...")
	faiss_index = FAISS.from_texts(chunks, embedding_model)

	if video_id:
		faiss_index.save_local(index_path)
		print(f"[+] FAISS index saved to: {index_path}")

	return faiss_index


def perform_similarity_search(faiss_index, query, k=3):
	print("[+] Performing Similarity Search ...")
	results = faiss_index.similarity_search(query, k=k)
	return results


def create_summary_prompt():
	print("[+] Creating summary prompt ...")
	template = """
	<|begin_of_text|><|start_header_id|>system<|end_header_id|>
	You are an AI assistant tasked with summarizing YouTube video transcripts. Provide concise, informative summaries that capture the main points of the video content.

	Instructions:
	1. Summarize the transcript in a single concise paragraph.
	2. Ignore any timestamps in your summary.
	3. Focus on the spoken content (Text) of the video.

	Note: In the transcript, "Text" refers to the spoken words in the video, and "start" indicates the timestamp when that part begins in the video.<|eot_id|><|start_header_id|>user<|end_header_id|>
	Please summarize the following YouTube video transcript:

	{transcript}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
	"""
	prompt = PromptTemplate(
		input_variables=["transcript"],
		template=template
	)
	return prompt


def create_summary_chain(llm, prompt):
	print("[+] Creating Summary Chain...")
	return prompt | llm | StrOutputParser()


def retrieve(query, faiss_index, k=7):
	print("[+] Retrieve Function...")
	relevant_context = faiss_index.similarity_search(query, k=k)
	return relevant_context


def create_qa_prompt_template():
	print("[+] Creating QA Prompt Template...")
	qa_template = """
	You are an expert assistant providing detailed answers based on the following video content.

	Relevant Video Context: {context}

	Based on the above context, please answer the following question:
	Question: {question}
	"""
	prompt_template = PromptTemplate(
		input_variables=["context", "question"],
		template=qa_template
	)
	return prompt_template


def create_qa_chain(llm, prompt_template, verbose=True):
	print("[+] Creating QA Chain ...")
	return prompt_template | llm | StrOutputParser()


def generate_answer(question, faiss_index, qa_chain, k=7):
	print("[+] Generate Answer ...")
	relevant_context = retrieve(question, faiss_index, k=k)
	context = "\n\n".join(doc.page_content for doc in relevant_context)
	answer = qa_chain.invoke({"context": context, "question": question})
	return answer