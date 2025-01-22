import os
import gradio as gr
import whisper
import librosa
import soundfile as sf
import tempfile
import pyttsx3  # Import for text-to-speech
import spacy

# Load English NER model from spaCy
nlp = spacy.load("en_core_web_md")

model = whisper.load_model("base")

def load_and_resample_audio(file_path, target_sample_rate=16000):
    audio, _ = librosa.load(file_path, sr=target_sample_rate)
    directory = os.path.join(os.path.dirname(__file__), "recorded_audio")
    os.makedirs(directory, exist_ok=True)
    temp_file_path = os.path.join(directory, 'resampled_audio.wav')
    sf.write(temp_file_path, audio, target_sample_rate)

    # Check if the file exists and has non-zero size
    if os.path.isfile(temp_file_path) and os.path.getsize(temp_file_path) > 0:
        return temp_file_path
    else:
        raise FileNotFoundError(f"Failed to create or save resampled audio file at: {temp_file_path}")

def extract_named_entities(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

def inference(audio_file_path):
    try:
        resampled_audio_path = load_and_resample_audio(audio_file_path)
        audio = whisper.load_audio(resampled_audio_path)
        audio = whisper.pad_or_trim(audio)

        mel = whisper.log_mel_spectrogram(audio).to(model.device)
        _, probs = model.detect_language(mel)
        lang = max(probs, key=probs.get)

        options = whisper.DecodingOptions(fp16=False)
        result = whisper.decode(model, mel, options)

        transcribed_text = result.text
        named_entities = extract_named_entities(transcribed_text)

        return lang.upper(), transcribed_text, named_entities, True

    except Exception as e:
        print(f"Error during inference: {e}")
        # Inform user about potential issues
        return None, "An error occurred during audio processing. Please try again.", [], False

def speak_text(text):
    try:
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        engine.setProperty('voice', voices[0].id)  # Use the first voice by default
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print(f"Error speaking text: {e}")
        # Inform user about potential issues
        return "Couldn't speak the text due to an error."

title = "Speech Transcription"
description = "Transcribe audio files or recordings using Whisper and speak the transcribed text."

block = gr.Blocks()

with block:
    gr.Markdown("# Speech to Text & Speak")
    with gr.Column():
        audio = gr.Audio(label="Input Audio", type="filepath")
        transcribe_btn = gr.Button("Transcribe & Speak")
        lang_str = gr.Textbox(label="Language")
        text = gr.Textbox(label="Transcription")
        entities_output = gr.Textbox(label="Named Entities")

    transcribe_btn.click(
    fn=lambda audio_file_path: (
        *inference(audio_file_path),
        speak_text(text.value) if text.value else "No transcription available."
    ),
    inputs=[audio],
    outputs=[lang_str, text, entities_output]
)

block.launch()
