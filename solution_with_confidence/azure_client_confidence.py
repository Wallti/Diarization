# client.py

import asyncio
import websockets
import json
import pyaudio
import numpy as np
from llm_validation_service import LLMValidationService

# WebSocket Server URL
WS_SERVER = "ws://localhost:8765"

# Audio Configuration
FORMAT = pyaudio.paInt16  # 16-bit PCM
CHANNELS = 1  # Mono
RATE = 44100  # 44.1kHz
CHUNK = 1024  # Audio chunk size
SILENT_CHUNK = (np.zeros(CHUNK, dtype=np.int16)).tobytes()
llm_validator = LLMValidationService()

# Connection Control Variable
continuous_connection = True  # Set to False to stop all connections
recent_confidences = []  # Track last few confidence scores

# Speaker configurations as dictionary entries
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
            print(f" Speaker {speaker_id} Ping failed: {e}")
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
            print(f" Speaker {speaker_id} connected to WebSocket server")

            # Start periodic ping task
            asyncio.create_task(send_ping(ws, speaker_id))

            # Send language selection message for each speaker
            language_message = json.dumps({
                "type": "language",
                "code": speakers[speaker_id]["language"],
                "speaker_id": speaker_id
            })
            await ws.send(language_message)
            print(f" Speaker {speaker_id} sent language selection to server")
        except Exception as e:
            print(f" Could not connect Speaker {speaker_id}: {e}")

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
                    print(f" Speaker {speaker_id} sent {len(data)} bytes of audio")
                except Exception as e:
                    print(f" Speaker {speaker_id} error: {e}")

    except Exception as e:
        print(f" Audio streaming error: {e}")

    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()
        for ws in websockets_dict.values():
            await ws.close()
        print(" Stopped all audio streams.")

def average_confidence(scores):
    return sum(scores) / len(scores) if scores else 0        

async def receive_transcriptions():
    """Receives transcriptions, validates with LLM, and filters out incorrect languages."""
    global continuous_connection

    while continuous_connection:
        try:
            async with websockets.connect(WS_SERVER, open_timeout=60) as websocket:
                print(" Connected to WebSocket server for transcriptions.")

                async for message in websocket:
                    try:
                        data = json.loads(message)

                        if data["type"] == "transcription":
                            speaker_id = data['speaker']
                            text = data['text']
                            detected_lang = speakers.get(speaker_id, {}).get("language", "unknown")
                            server_lang = data.get("detected_language", detected_lang)
                            azure_confidence = data.get("confidence", 0.0)  

                            if speaker_id not in recent_confidences:
                                recent_confidences[speaker_id] = []

                            # Add the current confidence score to the list
                            recent_confidences[speaker_id].append(azure_confidence)

                            # Keep only the last 5 scores for averaging
                            if len(recent_confidences[speaker_id]) > 5:
                                recent_confidences[speaker_id].pop(0)

                            # Calculate the average confidence score for the last few transcriptions
                            avg_confidence = average_confidence(recent_confidences[speaker_id])

                            # Dynamic threshold logic, you can adjust based on speaker or language
                            dynamic_threshold = 0.85
                            if azure_confidence < dynamic_threshold and avg_confidence < dynamic_threshold:
                                print(f" [LOW CONFIDENCE] Speaker {speaker_id}: {text} (Confidence: {azure_confidence}, Avg: {avg_confidence})")
                                continue


                            #  Reject low-confidence transcriptions
                            if azure_confidence >= 0.90:
                                print(f" High Confidence for Speaker {speaker_id}: {text}")
                                # Process with LLM validation
                            elif 0.80 <= azure_confidence < 0.90:
                                print(f" Medium Confidence for Speaker {speaker_id}: {text}")
                                # Flag for review or ask for human verification
                            else:
                                print(f" Low Confidence for Speaker {speaker_id}: {text}")
                                # Reject and move to next
                                continue
                            #  Reject if detected language does not match expected speaker language
                            if server_lang != detected_lang:
                                print(f" [LANGUAGE MISMATCH] Speaker {speaker_id}: {text}")
                                print(f"   - Expected: {detected_lang}, Got: {server_lang} (Confidence: {azure_confidence})")
                                continue  # Skip processing

                            #  Validate with LLM before finalizing
                            validation_result = await llm_validator.validate_text(text, detected_lang, "en", azure_confidence)

                            if validation_result and validation_result["is_valid"]:
                                validated_text = validation_result["original_text"]
                                print(f" {data['timestamp']} | Speaker {speaker_id}: {validated_text} (Confidence: {azure_confidence})")
                            else:
                                print(f" [INVALID] Speaker {speaker_id}: {text} - {validation_result.get('reasoning', 'Unknown error')}")

                    except json.JSONDecodeError:
                        print(f" Invalid JSON received: {message}")

        except websockets.exceptions.ConnectionClosedError as e:
            print(f" WebSocket closed unexpectedly: {e}. Reconnecting in 5s...")
            await asyncio.sleep(5)  # Retry connection

        except Exception as e:
            print(f" Unexpected error: {e}. Reconnecting in 5s...")
            await asyncio.sleep(5)  # Retry connection




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
        print("\n Stopping all connections...")
