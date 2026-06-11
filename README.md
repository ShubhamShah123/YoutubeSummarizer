# 🎬 YTBot — YouTube Video Summarizer & Q&A

A local AI-powered app that summarizes YouTube videos and answers questions about them — using Google Gemini for language, HuggingFace embeddings for semantic search, and FAISS for fast retrieval. Built with Gradio for a simple browser UI.

---

## Features

- **Summarize any YouTube video** from its transcript in seconds
- **Ask questions** about the video content using RAG (Retrieval-Augmented Generation)
- **Smart caching** — transcripts and FAISS indexes are saved to disk so nothing is recomputed on subsequent runs
- **Multilingual support** — handles English and Hindi transcripts

---

## Project Structure

```
ytbot/
├── ytbot.py                  # Main app — Gradio UI and button logic
├── preprocess_functions.py   # Transcript fetching, processing, chunking
├── setup_functions.py        # Model and embedding initialization (singleton)
├── FAISS_functions.py        # Vector index creation, retrieval, QA chain
├── .env                      # Your API keys (not committed)
├── requirements.txt          # Python dependencies
└── cache/                    # Auto-created; stores transcripts and FAISS indexes
    ├── <video_id>_transcript.txt
    └── <video_id>_faiss_index/
```

---

## Quickstart

### 1. Clone the repo

```bash
git clone https://github.com/your-username/ytbot.git
cd ytbot
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up your API key

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_google_gemini_api_key_here
```

Get a free Gemini API key at [aistudio.google.com](https://aistudio.google.com).

### 4. Run the app

```bash
python ytbot.py
```

Then open your browser at `http://localhost:7860`.

---

## How to Use

1. Paste a YouTube URL into the **YouTube Video URL** field
2. Click **Summarize Video** to get a concise summary of the video
3. Type a question in the **Ask a Question** field and click **Ask a Question** to get an answer based on the video content

> The first run for a given video fetches and indexes the transcript. Every subsequent run loads it instantly from the local cache.

---

## How Caching Works

| Data | Stored at | When |
|---|---|---|
| Processed transcript | `cache/<video_id>_transcript.txt` | After first fetch |
| FAISS vector index | `cache/<video_id>_faiss_index/` | After first Q&A |
| LLM & embedding model | In-memory singleton | Once per session |

To force a refresh for a video, delete its files from the `cache/` folder.

---

## Dependencies

| Library | Purpose |
|---|---|
| `gradio` | Browser UI |
| `youtube-transcript-api` | Fetching video transcripts |
| `langchain` | LLM chains and prompt templates |
| `langchain-google-genai` | Gemini model integration |
| `langchain-huggingface` | HuggingFace embedding model |
| `langchain-community` | FAISS vector store wrapper |
| `faiss-cpu` | Vector similarity search |
| `python-dotenv` | Loading `.env` API keys |

Install all at once:

```bash
pip install gradio youtube-transcript-api langchain langchain-google-genai \
            langchain-huggingface langchain-community faiss-cpu python-dotenv
```

---

## Models Used

- **LLM:** `gemini-2.5-flash-lite` via Google Generative AI
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` via HuggingFace

---

## Notes

- Only videos with available transcripts (auto-generated or manual) in **English or Hindi** are supported
- The `cache/` directory is created automatically on first run — you can safely add it to `.gitignore`
- The embedding model (~90MB) is downloaded once by HuggingFace and cached locally in `~/.cache/huggingface/`

---

## License

MIT
