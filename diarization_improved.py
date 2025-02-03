import os
import pyaudio
import numpy as np
import wave
import whisper
import torch
import logging
import azure.cognitiveservices.speech as speechsdk
from groq import Groq
from scipy.signal import butter, lfilter
from multiprocessing import Process, Queue


logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s: %(message)s')
# Configurations
WHISPER_LANGUAGE = "en"  # Change this to your target language
GROQ_API_KEY = "..."
AUDIO_FILENAME = "input_audio.wav"
STT_MODEL = "whisper-large-v3-turbo"
PROCESS_EVERY = 5  # Process every 5 chunks for lower latency
OVERLAP = 2  # Keep a slight overlap to avoid cut-off words



# Audio Streaming Settings
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 4096  # Initial value, will be dynamically adjusted

def adjust_chunk_size(audio_data):
    """Dynamically adjust CHUNK size based on audio intensity."""
    global CHUNK
    avg_amplitude = np.mean(np.abs(audio_data))
    
    if avg_amplitude < 500:  # If the signal is quiet, increase CHUNK
        CHUNK = min(CHUNK * 2, 8192)  
    elif avg_amplitude > 3000:  # If the signal is loud, decrease CHUNK
        CHUNK = max(CHUNK // 2, 1024)  
    
    return CHUNK

def highpass_filter(audio_data, cutoff=100, fs=16000, order=5):
    """Apply high-pass filter to remove low-frequency noise."""
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return lfilter(b, a, audio_data)

def normalize_audio(audio_data):
    """Normalize volume levels."""
    return audio_data / np.max(np.abs(audio_data))


def get_Groq_Transcription(audio_chunk_filename):
    """Transcribe audio and check for supported language."""
    client = Groq(api_key=GROQ_API_KEY)

    # Read the WAV file before sending it to Groq for transcription
    with open(audio_chunk_filename, 'rb') as audio_file:
        transcription = client.audio.transcriptions.create(
            file=("chunk.wav", audio_file.read()),
            model=STT_MODEL,
            response_format="json",
            temperature=0.0
        )

    print("Transcription Response:", transcription)
    transcription_text = transcription.text
    
    # Since no language is detected, return the transcription only
    return transcription_text, None, True

def transcribe_whisper(audio_file):
    try:
        transcription_text, _, _ = get_Groq_Transcription(audio_file)
        logging.info(f"Transcription: {transcription_text}")
        print(f"Transcription: {transcription_text}")
    except Exception as e:
        logging.error(f"Transcription error: {e}")
        print(f"Transcription error: {e}")


def create_azure_diarization_client(audio_file, speakers_config):
    """Advanced diarization with multi-language support"""

    speech_key = os.environ.get("SPEECH_KEY")
    speech_region = os.environ.get("SPEECH_REGION")
    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=speech_region)
    
    # Configure multiple speakers with their languages
    conversation_transcriber = speechsdk.transcription.ConversationTranscriber(speech_config)
    
    for speaker_id, language in speakers_config.items():
        participant = speechsdk.transcription.Participant(
            speech_config, 
            language=language,
            name=f"Speaker {speaker_id}"
        )
        conversation_transcriber.add_participant(participant)
    
    audio_config = speechsdk.audio.AudioConfig(filename=audio_file)
    conversation_transcriber.transcribed.connect(on_transcribed)
    conversation_transcriber.start_transcribing()

def on_transcribed(event):
    """Handle transcription for each speaker"""
    print(f"Speaker {event.result.speaker_id}: {event.result.text} (Language: {event.result.language})")

# Usage example
speakers_config = {
    1: "en-US",  # English speaker
    2: "es-ES",  # Spanish speaker
    3: "fr-FR"   # French speaker
}


def diarization_result_cb(event):
    """Process Azure diarization results."""
    if event.result.reason == speechsdk.ResultReason.RecognizedSpeech:
        print(f"Speaker {event.result.speaker_id}: {event.result.text}")

def match_diarization_with_whisper(whisper_segments, speaker_segments):
    """Match Whisper transcription with Azure speaker IDs."""
    output = []
    for w_segment in whisper_segments:
        w_text = w_segment['text']
        w_start = w_segment['start']
        speaker = None
        for s_speaker, s_text, s_start in speaker_segments:
            if abs(w_start - s_start) < 1.0:
                speaker = s_speaker
                break
        output.append(f"Speaker {speaker}: {w_text}" if speaker else w_text)
    return "\n".join(output)

def process_audio_buffer(buffer):
    """Process buffered audio: filter, transcribe with Whisper, and diarize with Azure."""
    audio_data = np.frombuffer(b''.join(buffer), dtype=np.int16)
    audio_data = highpass_filter(audio_data)
    audio_data = normalize_audio(audio_data)
    
    audio_file = "chunk.wav"
    with wave.open(audio_file, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(pyaudio.PyAudio().get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(audio_data.tobytes())
    
    whisper_proc = Process(target=transcribe_whisper, args=(audio_file,))
    azure_proc = Process(target=create_azure_diarization_client, args=(audio_file,))
    whisper_proc.start()
    azure_proc.start()
    whisper_proc.join()
    azure_proc.join()

def stream_audio_and_transcribe():
    """Continuously streams audio and sends chunks for real-time transcription."""
    global CHUNK  # Ensure CHUNK is modified globally
    
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, 
                frames_per_buffer=CHUNK)    # Add input latency
    buffer = []
    print("Streaming and transcribing... Press Ctrl+C to stop.")

    try:
        while True:
            try:
                audio_chunk = stream.read(CHUNK, exception_on_overflow=False)  # Handle overflow
                audio_data = np.frombuffer(audio_chunk, dtype=np.int16)
                
                # Adjust CHUNK dynamically
                CHUNK = adjust_chunk_size(audio_data)
                
                buffer.append(audio_chunk)
                if len(buffer) >= PROCESS_EVERY:
                    process_audio_buffer(buffer)
                    buffer = buffer[-OVERLAP:]  # Keep overlap buffer

            except OSError as e:
                print(f"Audio stream overflow: {e}")  # Catch buffer overflow errors
                buffer = []  # Reset buffer to avoid large overflows

    except KeyboardInterrupt:
        print("Stopping audio stream.")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

if __name__ == "__main__":
    stream_audio_and_transcribe()
