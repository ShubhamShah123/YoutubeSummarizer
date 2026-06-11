import re
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter

ytt_api = YouTubeTranscriptApi()

def get_video_id(url):    
	print("[+] get_video_url()")
	pattern = r'https:\/\/www\.youtube\.com\/watch\?v=([a-zA-Z0-9_-]{11})'
	match = re.search(pattern, url)
	return match.group(1) if match else None

def get_transcript(url):
	print("[+] get_transcript()")
	video_id = get_video_id(url)
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
	return transcript if transcript else None
	
tscript = get_transcript(input("URL: "))

print(tscript)