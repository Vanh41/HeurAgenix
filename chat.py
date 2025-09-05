import logging
import os
from src.util.llm_client.get_llm_client import get_llm_client

# config_file = os.path.join("output", "llm_config", "azure_gpt_o3.json")
# os.environ["GEMINI_API_KEY"] = "AIzaSyChjTXB3V89ddYfiHIVzjVfovvhYeQRDEs"
# ROOT_DIR = os.getcwd()
# logging.basicConfig(level=logging.INFO)

config_file = "llm_config.json"
llm_client = get_llm_client(config_file=config_file)
llm_client.load("Are you awake?")
response = llm_client.chat()
print(response)