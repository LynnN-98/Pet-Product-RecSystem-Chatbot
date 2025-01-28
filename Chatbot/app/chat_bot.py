import torch

def generate_answer(question, history, tokenizer, model):
    # Ensure `pad_token` is defined for tokenization
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))

    # Build conversation history from the last few turns
    MAX_HISTORY = 5  # Limit the number of conversation turns to include
    history_text = ""
    for turn in history[-MAX_HISTORY:]:
        history_text += f"User: {turn['user']}\nAssistant: {turn['assistant']}\n"
    prompt = history_text + f"User: {question}\nAssistant:"

    # Use GPU if available, otherwise default to CPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    # Tokenize the prompt and prepare attention mask
    inputs = tokenizer.encode_plus(
        prompt,
        return_tensors='pt',
        truncation=True,
        max_length=1024,  # Ensure input length stays within model capacity
        return_attention_mask=True
    )
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    # Generate a response with the model
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=input_ids.shape[1] + 150,  # Allow enough space for a complete response
            temperature=0.7,  # Control randomness (lower = focused, higher = creative)
            top_p=0.9,  # Nucleus sampling for diverse responses
            do_sample=True,
            num_beams=5,  # Use beam search for higher-quality results
            no_repeat_ngram_size=2,  # Avoid repetitive phrases
            early_stopping=True,  # Stop when the `eos_token` is reached
            eos_token_id=tokenizer.eos_token_id,  # End-of-sequence token
            pad_token_id=tokenizer.eos_token_id,  # Use `eos_token` for padding
            num_return_sequences=1  # Generate a single response
        )

    # Decode the output tokens into text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract the assistant's reply by removing the prompt
    reply = generated_text[len(prompt):].strip()

    # Post-process the reply to remove extra "User:" or "Assistant:" labels
    if "User:" in reply:
        reply = reply.split("User:")[0].strip()
    if "Assistant:" in reply:
        reply = reply.split("Assistant:")[0].strip()

    return reply  # Return the clean response
