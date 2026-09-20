from openai import OpenAI

OLLAMA_BASE_URL = "http://172.28.57.218:11434/v1"
LOCAL_MODEL_NAME = "qwen2.5:7b"

local_client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")