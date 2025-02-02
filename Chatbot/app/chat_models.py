import os
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from peft import PeftModel

def get_chat_model_loader():
    def load_chat_model():
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Adjust the path so MODELS_DIR is where your models_cli folder is located.
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        MODELS_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', 'models_cli'))
        chat_model_path = os.path.join(MODELS_DIR, 'gpt2')
        
        # Load the tokenizer (from the adapter folder) and set the pad token if needed.
        chat_tokenizer = GPT2Tokenizer.from_pretrained(chat_model_path)
        chat_tokenizer.pad_token = chat_tokenizer.eos_token  # Ensure the pad token is set
        
        # Load the base GPT-2 model from its original pretrained source.
        base_model = GPT2LMHeadModel.from_pretrained("gpt2")
        
        # ***IMPORTANT: Resize the embeddings to match the tokenizer's vocabulary.***
        base_model.resize_token_embeddings(len(chat_tokenizer))
        
        # Load the LoRA adapter onto the base model.
        chat_model = PeftModel.from_pretrained(base_model, chat_model_path)
        
        # Move the model to the appropriate device and set it to eval mode.
        chat_model.to(device)
        chat_model.eval()
        
        return chat_tokenizer, chat_model
    return load_chat_model
