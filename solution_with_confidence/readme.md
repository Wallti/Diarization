The client is an audio streaming application that captures microphone input, sends it to a WebSocket server for transcription, receives transcriptions, and validates them with an LLM-based confidence check. Below is a breakdown of how the different components interact.

1. Client-to-Server Interaction (Audio Streaming & Transcription)

a) Establishing WebSocket Connections
b) Sending Audio Data
c) Receiving Transcriptions (Each transcription message includes:
	                            •	The transcribed text
	                            •	The detected language
	                            •	The confidence score from Azure Speech API)

2. LLM Confidence Handling (Validation & Filtering)
   
a) Confidence Score Check (Azure)
	•	The client first checks the Azure confidence score (from the transcription API).
	•	If the confidence score is below 0.85, the transcription is discarded immediately.

b) Language Validation
	•	The client compares the expected speaker language with the detected language from the transcription.
	•	If there is a mismatch (e.g., a German speaker’s audio is transcribed as French), the transcription is discarded.

 c) LLM-Based Confidence Validation
	If the transcription passes the Azure confidence and language checks, it is further validated by the LLM.
	•	The client calls the LLM validation service (llm_validator.validate_text(...)).
	•	The LLM performs an additional layer of validation, checking for:
	•	Text correctness
	•	Context accuracy
	•	Language structure consistency

4. Integration of Confidence Averaging
	•	The client now keeps track of recent confidence scores per speaker.
	•	If individual scores fluctuate, it considers an average of the last 5 scores.
	•	This prevents small confidence drops from unnecessarily discarding a valid transcription.


 

