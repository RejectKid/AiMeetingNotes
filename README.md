# AiMeetingNotes

🎙️ **AiMeetingNotes** is a Python-based tool designed to record meetings, transcribe audio, and generate concise summaries. It captures audio in 2.5-minute intervals, transcribes the content using OpenAI's Whisper model, and summarizes the transcriptions with Facebook's BART model.

## Features

* **Live Audio Recording**: Captures audio in 2.5-minute chunks.
* **Transcription**: Utilizes OpenAI's Whisper model for accurate transcriptions.
* **Summarization**: Employs Facebook's BART model to generate concise summaries.
* **Timestamped Logs**: Saves both raw transcripts and summaries with timestamps.
* **GPU Support**: Leverages GPU acceleration if available for faster processing.

## Installation

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/RejectKid/AiMeetingNotes.git
   cd AiMeetingNotes
   ```

2. **Create a Virtual Environment** (Optional but recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the main script to start recording, transcribing, and summarizing:

```bash
python audio_to_text_notes.py
```

* The script will continuously record audio in 2.5-minute intervals.
* Transcriptions will be saved in `raw_transcripts.txt`.
* Summaries will be saved in `notes.txt`.

Press `Ctrl+C` to stop the recording.

## Requirements

* Python 3.7 or higher
* [sounddevice](https://pypi.org/project/sounddevice/)
* [numpy](https://pypi.org/project/numpy/)
* [whisper](https://pypi.org/project/whisper/)
* [transformers](https://pypi.org/project/transformers/)
* [torch](https://pypi.org/project/torch/)

Ensure that your system has a working microphone and the necessary permissions to access it.

## Configuration

* **Chunk Duration**: Adjust the `CHUNK_DURATION` variable in `audio_to_text_notes.py` to change the length of each audio recording segment.
* **Model Selection**: Modify the model names in the script if you wish to use different transcription or summarization models.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your changes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

Feel free to customize this `README.md` further to suit your project's specific needs.
