import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# CUDA_VISIBLE_DEVICES=6 python generate_unaligned.py

def extract_response(text):
    last_msg = text.rsplit('ASSISTANT:', 1)[-1].rsplit('<eos>', 1)[0].strip()
    return last_msg

def main(split: str):
    questions = pd.read_csv(f'./datasets/{split}.csv')['prompt'].tolist()
    initial_prompt = pd.read_csv(f'./datasets/{split}.csv')['target'].tolist()

    assert len(questions) == len(initial_prompt)
    # model_name = "TheBloke/Wizard-Vicuna-7B-Uncensored-GPTQ"
    model_name = "QuixiAI/Wizard-Vicuna-7B-Uncensored"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",  # Handles multi-GPU or CPU deployment
        trust_remote_code=True,  # Required for some models
        torch_dtype=torch.float16,  # Use FP16 precision
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")

    def generate_response(messages):
        tokenizer.chat_template = open("vicuna.jinja").read()
        formatted_messages = tokenizer.apply_chat_template(messages, tokenize=False, add_special_tokens=False, continue_final_message=True)
        inputs = tokenizer(formatted_messages, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_length=512, pad_token_id=tokenizer.eos_token_id)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    responses = []
    print(f"Generating responses for {split} dataset")
    print(f"Number of questions: {len(questions)}")

    count = 0
    for q, initial_p in zip(questions, initial_prompt):
        # prepare the prompt template
        messages = [
            {"role": "user", "content": q},
            {"role": "assistant", "content": initial_p}
        ]
        LLM_response = generate_response(messages)
        print("-" * 100)
        print(f"Count: {count}")
        print(f"Question: {q}")
        print(f"Initial prompt: {initial_p}")
        response = extract_response(LLM_response)
        print(f"Response: {response}")
        print("-" * 100)
        responses.append(response)
        count += 1
    # save the responses
    df = pd.DataFrame({'question': questions, 'response': responses})
    df.to_csv(f'./datasets/unalign_{split}.csv', index=False)

if __name__ == "__main__":
    
    #main("test")
    main("train")
    main("val")