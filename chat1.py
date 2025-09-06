import logging
import os
import traceback
from src.util.llm_client.get_llm_client import get_llm_client

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

config_file = "llm_config1.json"

try:
    llm_client = get_llm_client(config_file=config_file)
    print(f"LLM Client type: {type(llm_client)}")
    
    llm_client.load("Are you awake?")
    print("Message loaded successfully")
    
    response = llm_client.chat()
    print("Response:", response)
    
except Exception as e:
    print(f"Error: {e}")
    print("Full traceback:")
    traceback.print_exc()