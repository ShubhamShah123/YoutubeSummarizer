import re
import os
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter
print("[+] Preprocess Libraries imported!")

ytt_api = YouTubeTranscriptApi()

CACHE_DIR = "cache"


def get_video_id(url):
	print("[+] get_video_url()")
	pattern = r'https:\/\/www\.youtube\.com\/watch\?v=([a-zA-Z0-9_-]{11})'
	match = re.search(pattern, url)
	return match.group(1) if match else None


def get_transcript(url):
	print("[+] get_transcript()")
	video_id = get_video_id(url)
	if not video_id:
		return None

	# --- Cache check ---
	cache_path = os.path.join(CACHE_DIR, f"{video_id}_transcript.txt")
	if os.path.exists(cache_path):
		print(f"[+] Loading transcript from cache: {cache_path}")
		with open(cache_path, "r", encoding="utf-8") as f:
			return f.read()  # returns already-processed text string

	# --- Fetch from YouTube ---
	transcripts = ytt_api.list(video_id)
	transcript = ""
	for t in transcripts:
		if t.language_code in ['en', 'hi']:
			if t.is_generated:
				if len(transcript) == 0:
					transcript = t.fetch()
		else:
			transcript = t.fetch()
			break

	if not transcript:
		return None

	# Process and save to cache
	processed = process(transcript)
	with open(cache_path, "w", encoding="utf-8") as f:
		f.write(processed)
	print(f"[+] Transcript cached at: {cache_path}")
	return processed


def process(transcript):
	"""Converts raw transcript list to text. Skipped if loading from cache."""
	print("[+] process()")
	# If transcript is already a string (loaded from cache), return as-is
	if isinstance(transcript, str):
		return transcript
	txt = ""
	for i in transcript:
		try:
			txt += f"Text: {i.text}. Start: {i.start}\n"
		except KeyError as e:
			print(f"[-] Error: {e}")
	return txt


def chunk_transcript(processed_transcript, chunk_size=200, chunk_overlap=20):
	print("[+] chunk_transcript()")
	text_splitter = RecursiveCharacterTextSplitter(
		chunk_size=chunk_size,
		chunk_overlap=chunk_overlap
	)
	chunks = text_splitter.split_text(processed_transcript)
	return chunks