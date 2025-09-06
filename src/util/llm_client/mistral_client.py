import os
import json
import requests
from src.util.llm_client.base_llm_client import BaseLLMClient


class MistralClient(BaseLLMClient):
    def __init__(
            self,
            config: dict,
            prompt_dir: str=None,
            output_dir: str=None,
        ):
        super().__init__(config, prompt_dir, output_dir)

        self.url = config["url"]
        model = config["model"]
        stream = config.get("stream", False)
        top_p = config.get("top_p", 0.7)
        temperature = config.get("temperature", 0.95)
        max_tokens = config.get("max_tokens", 3200)
        api_key = config["api_key"]
        self.max_attempts = config.get("max_attempts", 50)
        self.sleep_time = config.get("sleep_time", 60)
        
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        self.payload = {
            "model": model,
            "stream": stream,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p
        }

    def reset(self, output_dir:str=None) -> None:
        self.messages = []
        if output_dir is not None:
            self.output_dir = output_dir
            os.makedirs(output_dir, exist_ok=True)

    def _convert_messages_for_mistral(self):
        """Convert messages from BaseLLMClient format to Mistral API format"""
        mistral_messages = []
        for message in self.messages:
            role = message["role"]
            content = ""
            
            # Extract text content from the message
            for item in message["content"]:
                if item["type"] == "text":
                    content += item["text"]
                # Note: Mistral API may not support images, so we skip image content
            
            mistral_messages.append({
                "role": role,
                "content": content
            })
        
        return mistral_messages

    def chat_once(self) -> str:
        # Convert messages to Mistral format
        mistral_messages = self._convert_messages_for_mistral()
        self.payload["messages"] = mistral_messages
        
        response = requests.request("POST", self.url, json=self.payload, headers=self.headers)
        
        # Debug: print response for troubleshooting
        print(f"Response status: {response.status_code}")
        print(f"Response text: {response.text}")
        
        response_data = json.loads(response.text)
        
        # Check if response has the expected structure
        if "choices" in response_data and len(response_data["choices"]) > 0:
            response_content = response_data["choices"][0]["message"]["content"]
        else:
            raise Exception(f"Unexpected response format: {response_data}")
        
        return response_content