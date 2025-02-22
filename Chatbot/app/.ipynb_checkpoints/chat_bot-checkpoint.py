import torch
import re

def generate_answer(question, history, tokenizer, model):
    # define pad_token
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))

    # build conversation history and limit to the last 3 turns
    MAX_HISTORY = 3
    history_text = ""
    for turn in history[-MAX_HISTORY:]:
        # use get() for safe access to keys
        user_text = turn.get('user', '')
        assistant_text = turn.get('assistant', '')
        history_text += f"User: {user_text}\nAssistant: {assistant_text}\n"
    
    prompt = history_text + f"User: {question}\nAssistant:"

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    # encode the prompt and generate the attention mask
    inputs = tokenizer.encode_plus(
        prompt,
        return_tensors='pt',
        truncation=True,
        max_length=1024,  # ensure the prompt does not exceed the model's maximum input length
        return_attention_mask=True
    )
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    # dynamically adjust the generation length
    max_new_tokens = 150
    max_length = min(input_ids.shape[1] + max_new_tokens, 1024)
    # avoid exceeding the model's maximum context window when combined with a long prompt

    # generate the response
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
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

    # decoding
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    reply = generated_text[len(prompt):].strip()

    # clean up the reply
    if "User:" in reply:
        reply = reply.split("User:")[0].strip()
    if "Assistant:" in reply:
        reply = reply.split("Assistant:")[0].strip()
    if not reply:
        reply = "Sorry, I couldn't generate a response."
    reply = re.sub(r'https?://\S+', '', reply)
    reply = re.sub(r'\[URL[^\]]*\]', '', reply)
    reply = re.sub(r'\s+', ' ', reply).strip()

    # clean up GPU memory
    if device == 'cuda':
        torch.cuda.empty_cache()

    return reply
