import logging
from groq import AsyncGroq
from typing import Optional, Dict
import json
import os
from dotenv import load_dotenv


GROQ_API_KEY = "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"


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('llm_validation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

load_dotenv()

class LLMValidationService:
    def __init__(self):
        self.client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))  # Secure API Key via env
        self.system_prompt = """
        You are a strict language detector and validator for a real-time multilingual speech recognition system.
        Your primary task is to verify if the text is PURELY in the detected language and reject any mixed-language or incorrect detections.
        ...
        """

    async def validate_text(
        self, 
        text: str, 
        detected_lang: str, 
        alt_lang: str, 
        azure_confidence: float
    ) -> Optional[Dict]:
        if azure_confidence < 0.85:
            logger.info(f"Rejected due to low Azure confidence: {azure_confidence}")
            return {
                "is_valid": False,
                "confidence_score": 0.0,
                "reasoning": "Azure confidence too low"
            }

        user_prompt = f"""Strictly check this text: "{text}"
Detected Language: {detected_lang}

Analysis Steps:
1. Is this text 100% in {detected_lang} (excluding proper names)?
2. Are there ANY words from other languages (except proper names)?
3. If pure {detected_lang}, is it understandable?
4. If yes to both, translate to {alt_lang}

Respond in JSON:
{{
    "is_valid": bool,  
    "confidence_score": float,  
    "original_text": string,  
    "translation": string,  
    "reasoning": string  
}}"""

        try:
            response = await self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={ "type": "json_object" }
            )

            result = json.loads(response.choices[0].message.content)
            logger.info(f"LLM Validation Result: {result}")
            return result

        except Exception as e:
            logger.error(f"LLM Analysis Error: {e}")
            return None