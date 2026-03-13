import os
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM

load_dotenv()
token = os.getenv("HUGGING_FACE_TOKEN")  
if not token:
    raise ValueError("No Hugging Face token found in environment variables.")

login(token)

model_id = "meta-llama/Llama-3.2-1B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)