

import asyncio
import websockets
import json
import pyaudio
import numpy as np

# WebSocket Server URL
WS_SERVER = "ws://localhost:8765"

# Audio Configuration
FORMAT = pyaudio.paInt16  # 16-bit PCM
CHANNELS = 1  # Mono
RATE = 44100  # 44.1kHz
CHUNK = 1024  # Audio chunk size
SILENT_CHUNK = (np.zeros(CHUNK, dtype=np.int16)).tobytes()


# Connection Control Variable
continuous_connection = True  # Set to False to stop all connections

# Speaker configurations
speakers = {
    1: {"language": "it-IT"},
    2: {"language": "de-DE"},
    3: {"language": "fr-FR"},
}

async def send_ping(websocket, speaker_id):
    """Send periodic pings to keep WebSocket connection alive."""
    while continuous_connection:
        try:
            await websocket.ping()
            await asyncio.sleep(20)  # Ping every 20 seconds
        except Exception as e:
            print(f"⚠️ Speaker {speaker_id} Ping failed: {e}")
            break

async def send_audio():
    """Reads audio from a single microphone and sends it to multiple WebSockets."""
    global continuous_connection

    # Initialize PyAudio and open a single shared stream
    audio = pyaudio.PyAudio()
    stream = audio.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK
    )

    # Create separate WebSocket connections for each speaker
    websockets_dict = {}
    for speaker_id in speakers.keys():
        try:
            ws = await websockets.connect(WS_SERVER, open_timeout=60, ping_interval=20, ping_timeout=30)
            websockets_dict[speaker_id] = ws
            print(f"🔵 Speaker {speaker_id} connected to WebSocket server")

            # Start periodic ping task
            asyncio.create_task(send_ping(ws, speaker_id))

            # Send language selection message for each speaker
            language_message = json.dumps({
                "type": "language",
                "code": speakers[speaker_id]["language"],
                "speaker_id": speaker_id
            })
            await ws.send(language_message)
            print(f"📨 Speaker {speaker_id} sent language selection to server")
        except Exception as e:
            print(f"⚠️ Could not connect Speaker {speaker_id}: {e}")

    try:
        while continuous_connection:
            data = stream.read(CHUNK, exception_on_overflow=False)

        # If no real audio data is detected, send silence
            if not data.strip(b'\x00'):  # Check if the chunk is mostly silent
                data = SILENT_CHUNK

        # Send the same audio data to all WebSockets
            for speaker_id, ws in websockets_dict.items():
                try:
                    await ws.send(data)
                    print(f"🎤 Speaker {speaker_id} sent {len(data)} bytes of audio")
                except Exception as e:
                    print(f"⚠️ Speaker {speaker_id} error: {e}")

    except Exception as e:
        print(f"⚠️ Audio streaming error: {e}")

    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()
        for ws in websockets_dict.values():
            await ws.close()
        print("🛑 Stopped all audio streams.")

async def receive_transcriptions():
    """Receives transcriptions from the server and displays them."""
    global continuous_connection

    while continuous_connection:
        try:
            async with websockets.connect(WS_SERVER, open_timeout=60) as websocket:
                print("Connected to WebSocket server for transcriptions.")

                async for message in websocket:
                    try:
                        data = json.loads(message)
                        if data["type"] == "transcription":
                            print(f"{data['timestamp']} | Speaker {data['speaker']}: {data['text']}")
                    except json.JSONDecodeError:
                        print(f"Invalid JSON received: {message}")

        except Exception as e:
            print(f"🔄 Reconnecting transcription in 5s due to: {e}")
            await asyncio.sleep(5)  # Wait before retrying

async def main():
    """Manages the audio sender and transcription receiver."""
    tasks = [
        asyncio.create_task(send_audio()),  # Send the single mic input to multiple WebSockets
        asyncio.create_task(receive_transcriptions())  # Handle incoming transcriptions
    ]
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        continuous_connection = False
        print("\n🛑 Stopping all connections...")

## Code from Waleed

###
###

# import logging
# from groq import AsyncGroq
# from typing import Optional, Dict
# import json
# import os
# from dotenv import load_dotenv

# # Configure logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.FileHandler('llm_validation.log'),
#         logging.StreamHandler()
#     ]
# )
# logger = logging.getLogger(__name__)

# load_dotenv()

# class LLMValidationService:
#     def __init__(self):
#         self.client = AsyncGroq(api_key="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")
#         self.system_prompt = """
#         You are a strict language detector and validator for a real-time multilingual speech recognition system.
# Your primary task is to verify if the text is PURELY in the detected language and reject any mixed-language or incorrect detections.

# CRITICAL RULES:

# 0. IF LANGUAGE MATCHES:
#    - Accept ANY understandable speech in that language
#    - Accept slang, casual speech, broken grammar, word order issues
#    - Only reject if completely incomprehensible
#    - Do not try to correct or improve the text
 
# 1. PURE LANGUAGE CHECK:
#    - Text must be ENTIRELY in the detected language (except for proper names)
#    - ANY mixing of languages = IMMEDIATE REJECT
#    - Even 1-2 words from another language = REJECT
#    - Only allow proper names from other languages

# 2. AUTO-DETECTION RULES:
#    - When given a language code (e.g., en-US, fr-FR, de-DE, etc.):
#      * First identify the key characteristics of that language
#      * Check if ALL words match that language's patterns and vocabulary
#      * Remember proper names are allowed from any language
#      * Reject if you spot ANY words from other languages

# Example of validation process:
# Text: "Hello my name is Walid"
# Detected: en-US
# Result: ✓ VALID (All words are English except proper name)

# Text: "Hallo my Name ist Walid"
# Detected: de-DE
# Result: ✗ INVALID (Contains English words "my", mixed language)

# 3. VALIDATION STEPS:
#    1. Identify target language from code
#    2. Check if ALL words match that language
#    3. Allow proper names from any language
#    4. Reject ANY mixed language content
#    5. If pure language, check if understandable
#    6. Only translate if all checks pass

# 4. PROPER NAMES:
#    - Allow proper names (people, places, brands)
#    - Everything else must be in detected language
   
   
 

# Examples for only few languages to just to give example what i want auto replicate for other languages:
# English (en-US):
#    ✓ "Hello, my name is Walid" (Pure English with proper name)
#    ✗ "Hallo, my name is Walid" ("Hallo" is German, reject)
#    ✗ "Hello, ich bin Walid" (Mixed English-German, reject)

# German (de-DE):
#    ✓ "Hallo, ich heiße Walid" (Pure German with proper name)
#    ✗ "Hallo, my Name ist Walid" (Contains English "my", reject)
#    ✗ "Hello, ich bin Walid" (Mixed German-English, reject)

# French (fr-FR):
#    ✓ "Bonjour, je m'appelle Walid" (Pure French with proper name)
#    ✗ "Bonjour, my name is Walid" (Mixed French-English, reject)
#    ✗ "Hello, je suis Walid" (Mixed languages, reject)

# Spanish (es-ES):
#    ✓ "Hola, me llamo Walid" (Pure Spanish with proper name)
#    ✗ "Hola, my name is Walid" (Mixed Spanish-English, reject)
#    ✗ "Hello, me llamo Walid" (Mixed languages, reject)

# """

#     async def validate_text(
#         self, 
#         text: str, 
#         detected_lang: str, 
#         alt_lang: str, 
#         azure_confidence: float
#     ) -> Optional[Dict]:
#         # Early return if Azure confidence is too low
#         if azure_confidence < 0.3:
#             logger.info(f"Rejected due to low Azure confidence: {azure_confidence}")
#             return {
#                 "is_valid": False,
#                 "confidence_score": 0.0,
#                 "reasoning": "Azure confidence too low"
#             }

#         user_prompt = f"""Strictly check this text: "{text}"
# Detected Language: {detected_lang}

# Analysis Steps:
# 1. Is this text 100% in {detected_lang} (excluding proper names)?
# 2. Are there ANY words from other languages (except proper names)?
# 3. If pure {detected_lang}, is it understandable?
# 4. If yes to both, translate to {alt_lang}

# Respond in JSON:
# {{
#     "is_valid": bool,  # true ONLY if 100% in detected language AND understandable
#     "confidence_score": float,  # 0 if wrong/mixed language, 7-10 if pure language
#     "original_text": string,  # keep exactly as is
#     "translation": string,  # only if is_valid true
#     "reasoning": string  # explain language check result
# }}"""


#         try:
#             response = await self.client.chat.completions.create(
#                 model="llama-3.3-70b-versatile",
#                 messages=[
#                     {"role": "system", "content": self.system_prompt},
#                     {"role": "user", "content": user_prompt}
#                 ],
#                 response_format={ "type": "json_object" }
#             )
            
#             result = json.loads(response.choices[0].message.content)
#             logger.info(f"LLM Validation Result: {result}")
#             return result

#         except Exception as e:
#             logger.error(f"LLM Analysis Error: {e}")
#             return None
