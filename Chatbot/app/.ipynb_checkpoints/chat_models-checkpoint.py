import os
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from peft import PeftModel

def get_chat_model_loader():
    def load_chat_model():
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # BASE_DIR is the directory where the current file is located.
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        MODELS_DIR = os.path.join(BASE_DIR, '..', 'models_cli')
        
        # Define the path to the fine-tuned model adapter files
        chat_model_path = os.path.join(MODELS_DIR, 'gpt2')
        
        # Load the tokenizer from the adapter folder
        chat_tokenizer = GPT2Tokenizer.from_pretrained(chat_model_path)
        chat_tokenizer.pad_token = chat_tokenizer.eos_token  # Set padding token
        
        # Load the base GPT-2 model from its pretrained source
        base_model = GPT2LMHeadModel.from_pretrained("gpt2")
        
        # Load the adapter weights onto the base model using PEFT
        chat_model = PeftModel.from_pretrained(base_model, chat_model_path)
        
        # Move model to the appropriate device and set to evaluation mode
        chat_model.to(device)
        chat_model.eval()
        
        return chat_tokenizer, chat_model
    return load_chat_model
