import requests
import json

# Test Mistral API directly
url = "https://api.mistral.ai/v1/chat/completions"
headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer ews8MB0hZ8mSpbrXKMv5tRTDnfqs79iA"
}
data = {
    "model": "mistral-large-latest",
    "messages": [{"role": "user", "content": "Are you awake?"}],
    "temperature": 0.7,
    "max_tokens": 2048,
    "top_p": 1
}

response = requests.post(url, headers=headers, json=data)
print("Status Code:", response.status_code)
print("Response:", response.json())