import sounddevice as sd
import numpy as np
import whisper
import queue
import threading
import time
from transformers import pipeline
from datetime import datetime
from pyAudioAnalysis import audioSegmentation as aS
import soundfile as sf

# Load Whisper model on CPU
model = whisper.load_model("base")  # or "medium" for better accuracy

# Load summarization pipeline (CPU only)
summarizer = pipeline(
    "summarization",
    model="facebook/bart-large-cnn",
    device=-1
)

# Audio settings
SAMPLE_RATE = 16000
BLOCK_DURATION = 15  # x seconds
CHUNK_SIZE = int(SAMPLE_RATE * BLOCK_DURATION)

audio_queue = queue.Queue()

# Function to get the current time stamp
def get_current_timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def audio_callback(indata, frames, time_info, status):
    audio_queue.put(indata.copy())

def record_audio():
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", callback=audio_callback):
        print("🎙️ Recording... (Press Ctrl+C to stop)")
        while True:
            time.sleep(1)

def save_raw_transcript(text: str):
    timestamp = get_current_timestamp()
    with open("raw_transcripts.txt", "a", encoding="utf-8") as f:
        f.write(f"{timestamp} - {text.strip()}\n\n")

def save_summary_to_file(summary: str):
    timestamp = get_current_timestamp()
    with open("notes.txt", "a", encoding="utf-8") as f:
        f.write(f"{timestamp} - {summary.strip()}\n\n")

def perform_diarization(audio_file):
    # Perform speaker diarization
    print("Performing speaker diarization...")
    diarization_result = aS.speaker_diarization(audio_file, n_speakers=2)  # Adjust number of speakers if needed
    return diarization_result

def transcribe_and_summarize():
    buffer = []
    while True:
        try:
            data = audio_queue.get(timeout=10)
            buffer.append(data)

            if len(buffer) * data.shape[0] >= CHUNK_SIZE:
                audio_chunk = np.concatenate(buffer, axis=0)
                buffer = []

                print("📝 Transcribing chunk...")
                audio_np = audio_chunk.flatten()
                result = model.transcribe(audio_np, language="en")
                transcription = result["text"].strip()

                if len(transcription) > 20:
                    print("📄 Transcript:", transcription[:200] + "..." if len(transcription) > 200 else transcription)
                    save_raw_transcript(transcription)

                    # Save audio file temporarily for diarization
                    temp_audio_file = "temp_audio.wav"
                    sf.write(temp_audio_file, audio_chunk, SAMPLE_RATE)

                    # Perform speaker diarization
                    diarization_result = perform_diarization(temp_audio_file)

                    # Optionally, process the diarization_result to label speakers and transcribe
                    print("Diarization Result:", diarization_result)

                    print("✍️ Summarizing...")
                    summary = summarizer(
                        transcription,
                        max_length=150,  # 25*x sentences
                        min_length=75,  # 25*x sentences
                        do_sample=False
                    )[0]["summary_text"]

                    print("✅ Summary:\n", summary, "\n")
                    save_summary_to_file(summary)
        except queue.Empty:
            continue

# Start threads
record_thread = threading.Thread(target=record_audio, daemon=True)
process_thread = threading.Thread(target=transcribe_and_summarize, daemon=True)

record_thread.start()
process_thread.start()

# Keep the program alive
try:
    while True:
        time.sleep(10)
except KeyboardInterrupt:
    print("🛑 Exiting...")
