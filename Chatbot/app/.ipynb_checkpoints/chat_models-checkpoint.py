import os
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

def get_chat_model_loader():
    def load_chat_model():
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        MODELS_DIR = os.path.join(BASE_DIR, '../models_cli')
        
        # load the GPT-2 model after fine tuning
        chat_model_path = os.path.join(MODELS_DIR, 'gpt2')
        chat_tokenizer = GPT2Tokenizer.from_pretrained(chat_model_path)
        chat_model = GPT2LMHeadModel.from_pretrained(chat_model_path)
        chat_model.to(device)
        chat_model.eval()
        
        return chat_tokenizer, chat_model
    return load_chat_model
