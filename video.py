import spacy
from youtube_transcript_api import YouTubeTranscriptApi
import gradio as gr

NER = spacy.load("en_core_web_lg")

def get_transcript(youtube_video_url):
    video_id = youtube_video_url.split("=")[-1]  # Extract video ID from the URL
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        transcript_text = '\n'.join([line['text'] for line in transcript_list])
        return transcript_text
    except Exception as e:
        print(f"Error: {e}")
        return None

def analyze_entities(video_url):
    transcript = get_transcript(video_url)
    if transcript:
        text = NER(transcript)
        entities = [(ent.text, ent.label_) for ent in text.ents]
        return entities
    else:
        return [("Transcript not available.", "ERROR")]

# Create Gradio Interface
iface = gr.Interface(
    fn=analyze_entities,
    inputs=gr.Textbox(lines=1, label="Enter YouTube Video URL"),
    outputs=gr.Textbox(label="Extracted Entities"),
    title="YouTube Video NER",
    #description="Extract named entities (entities like persons, organizations, locations, etc.) from YouTube video transcripts.",
    theme="light"
)

iface.launch()
