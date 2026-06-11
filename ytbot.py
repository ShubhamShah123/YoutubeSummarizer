import gradio as gr
from preprocess_functions import *
from setup_functions import *
from FAISS_functions import *


def summarize_video(video_url):
	print("[+] Summarize Video...")
	global fetched_transcript, processed_transcript

	if video_url:
		fetched_transcript = get_transcript(video_url)
		processed_transcript = process(fetched_transcript)
	else:
		return "[-] Please provide a valid YouTube URL."

	if processed_transcript:
		model_id = setup_credentials()
		llm = setup_model(model_id)
		summary_prompt = create_summary_prompt()
		summary_chain = create_summary_chain(llm, summary_prompt)

		summary = summary_chain.invoke(
			{
				"transcript": processed_transcript
			}
		)
		return summary
	else:
		return "[-] No Summary Available. Please fetch the transcripts first."
	

def answer_question(video_url, user_question):
	global fetched_transcript, processed_transcript
	video_id = get_video_id(video_url)
	if not processed_transcript:
		if video_url:
			fetched_transcript = get_transcript(video_url)
			processed_transcript = process(fetched_transcript)
		else:
			return "[-] Please provide a valid YouTube URL."
	
	if processed_transcript and user_question:
		chunks = chunk_transcript(processed_transcript)
		model_id = setup_credentials()
		llm = setup_model(model_id)
		embedding_model = setup_embedding()
		faiss_index = create_faiss_index(chunks, embedding_model,video_id=video_id)

		qa_prompt = create_qa_prompt_template()
		qa_chain = create_qa_chain(llm, qa_prompt)

		answer = generate_answer(user_question, faiss_index, qa_chain)

		return answer
	else:
		return "[-] Please provide a valid question and ensure the transcript is fetched."

with gr.Blocks() as interface:
	# Input field for YouTube URL
	video_url = gr.Textbox(label="YouTube Video URL", placeholder="Enter the YouTube Video URL")
	
	# Outputs for summary and answer
	summary_output = gr.Textbox(label="Video Summary", lines=5)
	question_input = gr.Textbox(label="Ask a Question About the Video", placeholder="Ask your question")
	answer_output = gr.Textbox(label="Answer to Your Question", lines=5)

	# Buttons for selecting functionalities after fetching transcript
	summarize_btn = gr.Button("Summarize Video")
	question_btn = gr.Button("Ask a Question")

	# Display status message for transcript fetch
	transcript_status = gr.Textbox(label="Transcript Status", interactive=False)

	# Set up button actions
	summarize_btn.click(summarize_video, inputs=video_url, outputs=summary_output)
	question_btn.click(answer_question, inputs=[video_url, question_input], outputs=answer_output)

# Launch the app with specified server name and port
interface.launch(server_name="0.0.0.0", server_port=7860)
