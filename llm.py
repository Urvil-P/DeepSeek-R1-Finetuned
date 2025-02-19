from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

model_path = os.path.abspath(r"deepseek_finetuned\Finetuned")

base_tokenizer = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
tokenizer = AutoTokenizer.from_pretrained(base_tokenizer)

# Ensure PAD token is set if missing
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load the model correctly
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n USING DEVICE: {device} \n")
model = AutoModelForCausalLM.from_pretrained(
    model_path, 
    torch_dtype=torch.float32, #float16 not for cpu,    
).to(device)
# print("Model and Tokenizer Loaded Successfully!")


def Generate_Response(question, max_length=512, temperature=0.7, top_p=0.9):
    input_text = f"Question: {question}\nAnswer:"
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
    
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids, 
            attention_mask=attention_mask,  
            max_length=max_length, 
            temperature=temperature, 
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id  # Ensure proper padding
        )
    
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return response

if __name__ == "__main__":
    import time
    start = time.time()
    print(Generate_Response("What is Arjuna's dilemma?"))
    print("Total Time Taken:",time.time()-start)