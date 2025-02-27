# server.py
import asyncio
import websockets
import logging
import os
from datetime import datetime
from pydub import AudioSegment
import io
import wave
import numpy as np
import azure.cognitiveservices.speech as speechsdk
from dotenv import load_dotenv
import json
from pydub import AudioSegment


logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load Azure credentials
load_dotenv()
SPEECH_KEY = "b5b3434f17b34bc8acfc62d68f3cd687"
SPEECH_REGION = "francecentral"

# Supported languages
SUPPORTED_LANGUAGES = {
    "Danish": "da-DK",
    "German": "de-DE",
    "English (Australia)": "en-AU",
    "English (Canada)": "en-CA",
    "English (UK)": "en-GB",
    "English (Hong Kong)": "en-HK",
    "English (Ireland)": "en-IE",
    "English (India)": "en-IN",
    "English (Nigeria)": "en-NG",
    "English (New Zealand)": "en-NZ",
    "English (Philippines)": "en-PH",
    "English (Singapore)": "en-SG",
    "English (US)": "en-US",
    "Spanish (Spain)": "es-ES",
    "Spanish (Mexico)": "es-MX",
    "Finnish": "fi-FI",
    "French (Canada)": "fr-CA",
    "French (France)": "fr-FR",
    "Hindi": "hi-IN",
    "Italian": "it-IT",
    "Japanese": "ja-JP",
    "Korean": "ko-KR",
    "Norwegian": "nb-NO",
    "Dutch": "nl-NL",
    "Polish": "pl-PL",
    "Portuguese (Brazil)": "pt-BR",
    "Portuguese (Portugal)": "pt-PT",
    "Swedish": "sv-SE",
    "Turkish": "tr-TR",
    "Chinese (Mainland)": "zh-CN",
    "Chinese (Hong Kong)": "zh-HK"
}

def create_speech_recognizer(language_code, websocket):
    """Initialize Azure Speech Recognizer with real-time streaming"""

    speech_config = speechsdk.SpeechConfig(subscription=SPEECH_KEY, region=SPEECH_REGION)
    speech_config.speech_recognition_language = language_code
    
    # Enable speaker recognition (optional)
    speech_config.set_property(
        property_id=speechsdk.PropertyId.SpeechServiceResponse_DiarizeIntermediateResults,
        value='true'
    )

    # Audio Stream Format (16-bit PCM, 44.1kHz, mono)
    audio_format = speechsdk.audio.AudioStreamFormat(samples_per_second=44100, bits_per_sample=16, channels=1)
    stream = speechsdk.audio.PushAudioInputStream(audio_format)
    audio_config = speechsdk.audio.AudioConfig(stream=stream)

    recognizer = speechsdk.transcription.ConversationTranscriber(speech_config=speech_config, audio_config=audio_config)
    loop = asyncio.get_event_loop()

    async def send_transcription(text, speaker_id, status="recognized"):
        """Send transcription result to the WebSocket client"""
        try:
            message = json.dumps({
                "type": "transcription",
                "status": status,
                "text": text,
                "speaker": speaker_id,
                "timestamp": datetime.now().strftime("%H:%M:%S")
            })
            await websocket.send(message)
        except Exception as e:
            logger.error(f"Error sending transcription: {e}")

    def transcribed_cb(evt):
        """Handle final transcriptions"""
        if evt.result.text:
            logger.info(f"Recognized: {evt.result.text} (Speaker: {evt.result.speaker_id})")
            asyncio.run_coroutine_threadsafe(send_transcription(evt.result.text, evt.result.speaker_id, "recognized"), loop)

    def transcribing_cb(evt):
        """Handle intermediate transcriptions"""
        if evt.result.text:
            logger.info(f"Processing: {evt.result.text} (Speaker: {evt.result.speaker_id})")
            asyncio.run_coroutine_threadsafe(send_transcription(evt.result.text, evt.result.speaker_id, "processing"), loop)

    def canceled_cb(evt):
        """Handle errors or cancellation"""
        logger.error(f"STT Canceled: {evt.cancellation_details.reason}")
        if evt.cancellation_details.reason == speechsdk.CancellationReason.Error:
            asyncio.run_coroutine_threadsafe(send_transcription("Azure STT error occurred", "Unknown", "error"), loop)

    # Connect events
    recognizer.transcribed.connect(transcribed_cb)
    recognizer.transcribing.connect(transcribing_cb)
    recognizer.canceled.connect(canceled_cb)

    return recognizer, stream

def convert_pcm_to_audiosegment(pcm_data):
    """Convert raw PCM audio to a Pydub AudioSegment"""
    return AudioSegment.from_raw(io.BytesIO(pcm_data), sample_width=2, frame_rate=44100, channels=1)

async def handle_audio(websocket):
    """WebSocket connection handler"""
    client_id = id(websocket)
    logger.info(f"Client connected [ID: {client_id}]")

    speech_recognizer = None
    azure_stream = None
    selected_language = None

    try:
        async for message in websocket:
            if isinstance(message, str):  # JSON messages (language selection, stop command)
                try:
                    data = json.loads(message)
                    if data.get("type") == "language":
                        selected_language = data.get("code")
                        speech_recognizer, azure_stream = create_speech_recognizer(selected_language, websocket)
                        speech_recognizer.start_transcribing_async()
                        logger.info(f"Started transcription for {selected_language}")

                    elif data.get("type") == "stop":
                        logger.info(f"Stopping transcription for client {client_id}")
                        break

                except json.JSONDecodeError:
                    logger.error("Invalid JSON message")
                    continue

            elif isinstance(message, bytes) and speech_recognizer:
                # Convert PCM audio to Pydub format (optional, useful for processing)
                audio_segment = convert_pcm_to_audiosegment(message)

                # Send raw PCM to Azure STT immediately
                azure_stream.write(message)
                logger.info(f"Processed audio chunk: {len(message)} bytes")

    except websockets.exceptions.ConnectionClosedError:
        logger.warning(f"Client {client_id} disconnected abruptly.")

    except Exception as e:
        logger.error(f"Unexpected error: {e}")

    finally:
        if speech_recognizer:
            speech_recognizer.stop_transcribing_async()
            logger.info(f"Stopped transcription for client {client_id}")

async def main():
    server = await websockets.serve(handle_audio, "localhost", 8765)
    logger.info("WebSocket server started on ws://localhost:8765")
    await asyncio.Future()  # Keep the server running

if __name__ == "__main__":
    asyncio.run(main())