import torch
import torch

def generate_answer(question, history, tokenizer, model):
    # Ensure `pad_token` is defined for tokenization
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))

    # Build conversation history from the last few turns
    MAX_HISTORY = 5  # Limit how many prior turns to include in each prompt
    history_text = ""
    for turn in history[-MAX_HISTORY:]:
        history_text += f"User: {turn['user']}\nAssistant: {turn['assistant']}\n"
    prompt = history_text + f"User: {question}\nAssistant:"

    # Use GPU if available; otherwise CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # 1) Tokenize the prompt, truncating to avoid exceeding model capacity (1024 for base GPT-2)
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024,         # Clip the prompt if it’s longer than 1024 tokens
        return_attention_mask=True
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    # 2) Generate the response with `max_new_tokens` instead of a huge `max_length` 
    #    to stay within the context window
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=150,        # Limit how many tokens to generate
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            num_beams=5,
            no_repeat_ngram_size=2,
            early_stopping=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            num_return_sequences=1
        )

    # Decode the output tokens into text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 3) Extract just the assistant's reply by removing the prompt portion
    reply = generated_text[len(prompt):].strip()

    # 4) Remove any leftover "User:" or "Assistant:" from the end
    if "User:" in reply:
        reply = reply.split("User:")[0].strip()
    if "Assistant:" in reply:
        reply = reply.split("Assistant:")[0].strip()

    return reply
