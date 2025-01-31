import os
import azure.cognitiveservices.speech as speechsdk
import torch
import pyaudio
import numpy as np
import whisper

# Initialize Whisper model
device = "cuda" if torch.cuda.is_available() else "cpu"
whisper_model = whisper.load_model("medium").to(device)
print(f"Using device: {device}")

# PyAudio settings
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000  # 16kHz for Whisper
CHUNK = 1024

# Azure configuration for diarization
def create_azure_diarization_client():
    # Fetch Azure Speech API key and region from environment variables
    speech_key = os.environ.get("SPEECH_KEY")
    speech_region = os.environ.get("SPEECH_REGION")

    if not speech_key or not speech_region:
        raise ValueError("Azure Speech API key or region is not set in environment variables.")

    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=speech_region)
    speech_config.set_property(
        property_id=speechsdk.PropertyId.SpeechServiceResponse_DiarizeIntermediateResults,
        value="true"
    )
    audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
    return speechsdk.transcription.ConversationTranscriber(speech_config=speech_config, audio_config=audio_config)

# Function to extract speaker IDs and timestamps from Azure
def process_azure_diarization(conversation_transcriber, audio_data):
    diarization_results = []
    
    # Event handler to capture diarization results
    def diarization_transcribed_cb(evt: speechsdk.SpeechRecognitionEventArgs):
        if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
            diarization_results.append({
                "speaker_id": evt.result.speaker_id,
                "offset": evt.result.offset,
                "duration": evt.result.duration,
            })
    
    # Connect event handler
    conversation_transcriber.transcribed.connect(diarization_transcribed_cb)

    # Perform diarization on the given audio data
    stream = speechsdk.audio.PushAudioInputStream()
    audio_config = speechsdk.audio.AudioConfig(stream=stream)
    conversation_transcriber.audio_config = audio_config

    # Push audio data into Azure
    conversation_transcriber.start_transcribing_async()
    stream.write(audio_data.tobytes())
    stream.close()
    conversation_transcriber.stop_transcribing_async().get()

    return diarization_results

# Function to transcribe and assign speaker IDs
def whisper_with_azure_diarization():
    audio_interface = pyaudio.PyAudio()
    conversation_transcriber = create_azure_diarization_client()
    
    # Open microphone stream
    stream = audio_interface.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK
    )
    
    print("Starting transcription with Whisper and Azure diarization...")
    audio_buffer = np.empty((0,), dtype=np.int16)

    try:
        while True:
            audio_data = stream.read(CHUNK, exception_on_overflow=False)
            audio_samples = np.frombuffer(audio_data, dtype=np.int16)
            audio_buffer = np.concatenate((audio_buffer, audio_samples))

            # Process audio in ~5-second chunks (80,000 samples at 16kHz)
            if len(audio_buffer) > RATE * 5:
                # Diarization (Azure)
                diarization_results = process_azure_diarization(conversation_transcriber, audio_buffer[:RATE * 5])

                # Transcription (Whisper)
                audio_input = torch.from_numpy(audio_buffer[:RATE * 5]).float() / 32768.0
                whisper_result = whisper_model.transcribe(audio=audio_input, task="transcribe", language=None)  # language=None to auto-detect
                text = whisper_result.get("text", "").strip()

                # Match speaker ID with transcription
                if diarization_results:
                    speaker_info = diarization_results.pop(0)  # Get the first diarization result
                    print(f"\nSpeaker {speaker_info['speaker_id']} said: {text}")
                else:
                    print(f"\n(Unlabeled) {text}")

                # Remove processed audio
                audio_buffer = audio_buffer[RATE * 5:]
    except KeyboardInterrupt:
        print("\nStopping transcription...")
    finally:
        stream.stop_stream()
        stream.close()
        audio_interface.terminate()

# Run the transcription
if __name__ == "__main__":
    whisper_with_azure_diarization()