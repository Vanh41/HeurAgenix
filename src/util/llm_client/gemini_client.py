import os
import json
import requests
from src.util.llm_client.base_llm_client import BaseLLMClient


class GeminiClient(BaseLLMClient):
    def __init__(
            self,
            config: dict,
            prompt_dir: str=None,
            output_dir: str=None,
        ):
        super().__init__(config, prompt_dir, output_dir)

        self.base_url = config["url"]
        self.model = config["model"]
        self.api_key = config["api_key"]
        self.temperature = config.get("temperature", 0.7)
        self.top_p = config.get("top-p", 0.95)
        self.max_tokens = config.get("max_tokens", 3200)
        self.max_attempts = config.get("max_attempts", 50)
        self.sleep_time = config.get("sleep_time", 10)
        
        self.headers = {
            "Content-Type": "application/json",
        }

    def reset(self, output_dir: str = None) -> None:
        self.messages = []
        if output_dir is not None:
            self.output_dir = output_dir
            os.makedirs(output_dir, exist_ok=True)

    def chat_once(self) -> str:
        # Convert messages to Gemini format
        contents = []
        for message in self.messages:
            # Extract text content from the BaseLLMClient format
            text_content = ""
            if "content" in message and isinstance(message["content"], list):
                for content_item in message["content"]:
                    if content_item.get("type") == "text":
                        text_content += content_item.get("text", "")
                    # Skip image content for now (could be added later)
            else:
                # Fallback for any other format
                text_content = str(message.get("content", ""))
            
            if message["role"] == "user":
                contents.append({
                    "role": "user",
                    "parts": [{"text": text_content}]
                })
            elif message["role"] == "assistant":
                contents.append({
                    "role": "model", 
                    "parts": [{"text": text_content}]
                })

        url = f"{self.base_url}/models/{self.model}:generateContent"
        
        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": self.temperature,
                "topP": self.top_p,
                "maxOutputTokens": self.max_tokens,
            }
        }

        response = requests.post(f"{url}?key={self.api_key}", headers=self.headers, json=payload)
        
        if response.status_code != 200:
            raise Exception(f"API error: {response.status_code}, {response.text}")
            
        response_data = json.loads(response.text)
        
        if "candidates" not in response_data or not response_data["candidates"]:
            raise Exception(f"No candidates in response: {response.text}")
            
        content = response_data["candidates"][0]["content"]["parts"][0]["text"]
        return content
