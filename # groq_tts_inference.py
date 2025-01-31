#groq_tts_inference.py

import pyaudio
import wave
import numpy as np


from groq import Groq
from whisper_voice_list import DEFAULT_COUNTRY_CODES as whisper_DEFAULT_COUNTRY_CODES



STT_MODEL    = "whisper-large-v3"
GROQ_API_KEY = "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"
AUDIO_FILENAME = "input_audio.wav"
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 4096  # Adjust this value for latency/processing speed balance


def is_language_supported(language_logoi_isocode: str) -> bool:
    """Check if the detected language is supported."""
    return language_logoi_isocode in get_logoistandard_languages_list()

def get_logoistandard_languages_list():
    """Return a list of supported ISO language codes."""
    logoistandard_languages_list = []
    for isowhispercode in whisper_DEFAULT_COUNTRY_CODES:
        logoistandard_languages_list.append(isowhispercode)
    return logoistandard_languages_list

def get_Groq_Transcription(audio_chunk_filename):
    """Transcribe audio and check for supported language."""
    client = Groq(api_key=GROQ_API_KEY)

    # Read the WAV file before sending it to Groq for transcription
    with open(audio_chunk_filename, 'rb') as audio_file:
        transcription = client.audio.transcriptions.create(
            file=("chunk.wav", audio_file.read()),  # Make sure the correct file format is used
            model=STT_MODEL,
            response_format="json",
            temperature=0.0
        )

    # Print the entire transcription response to inspect its structure
    print("Transcription Response:", transcription)

    transcription_text = transcription.text

    # Since no language is detected, return the transcription only
    return transcription_text, None, True

def stream_audio_and_transcribe():
    """Continuously streams audio and sends chunks for real-time transcription."""
    p = pyaudio.PyAudio()

    # Open audio stream
    stream = p.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,
                    frames_per_buffer=CHUNK)

    print("Streaming and transcribing audio... Press Ctrl+C to stop.")
    buffer = []

    try:
        while True:
            try:
                # Read audio chunk from the stream
                audio_chunk = stream.read(CHUNK)

                # Save the chunk in the buffer
                buffer.append(audio_chunk)

                # Process after collecting enough audio (optional: chunk overlap for context)
                if len(buffer) >= 15:  # Adjust this value for latency/processing speed balance
                    process_audio_buffer(buffer)
                    buffer = []  # Clear the buffer after processing

            except OSError as e:
                # Handle PyAudio overflow (input buffer overflow)
                print(f"Audio stream error: {e}")
                buffer = []  # Reset buffer to avoid large overflows

    except KeyboardInterrupt:
        print("Stopping audio stream.")

    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

def process_audio_buffer(buffer):
    """Convert buffered audio to a valid .wav file and transcribe it."""
    # Convert the buffer to numpy array
    audio_data = np.frombuffer(b''.join(buffer), dtype=np.int16)

    # Save buffer as .wav file (correct file format and headers)
    audio_file = "chunk.wav"
    with wave.open(audio_file, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(pyaudio.PyAudio().get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(audio_data.tobytes())

    # Get transcription without language detection
    transcription, _, _ = get_Groq_Transcription(audio_file)

    print("Interim Transcription:", transcription)

# Main function
if __name__ == "__main__":
    stream_audio_and_transcribe()
