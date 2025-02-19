# Fine-tuned DeepSeek-R1-Distill-Qwen-1.5B on Bhagwad Geeta

## 📌 Project Overview
This is a basic **LLM Fine-tuning Implementation**, where the **DeepSeek-R1-Distill-Qwen-1.5B** model has been fine-tuned on **Bhagavad Gita** data. The dataset consists of **question-answer pairs** generated using ChatGPT from the teachings of Bhagavad Gita.

## 📖 Model Details
- **Base Model:** [DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B)
- **Dataset:** Question-Answer pairs  from Bhagavad Gita generated using ChatGPT
- **Fine-Tuning Method:** LoRA, PEFT (Parameter-Efficient Fine-Tuning)

## 🚀 Usage Instructions
To use the fine-tuned model, follow these steps:

### 1️⃣ Install Dependencies
```bash
pip install torch transformers peft
```

### 2️⃣ Load the Model
Coming Soon
<!-- ```python
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Define Model Path
model_path = os.path.abspath(r"deepseek_finetuned/Finetuned")

# Load Tokenizer
base_tokenizer = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
tokenizer = AutoTokenizer.from_pretrained(base_tokenizer)

# Ensure PAD token is set if missing
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load the Model
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n USING DEVICE: {device} \n")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float32,  # Use float16 only for GPU
).to(device) 
``` -->

### 3️⃣ Generate Responses
Coming Soon
<!-- ```python
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
``` -->

<!-- ### 4️⃣ Example Usage
```python
question = "What is the essence of Karma Yoga?"
response = Generate_Response(question)
print(response)
``` -->

## ⚡ Future Improvements
- Expanding the dataset with more **diverse and detailed** question-answer pairs.
- Fine-tuning with **better optimization techniques**.
- Deploying the model as an **API or chatbot** for easy access.


## 📬 Contact
For any questions or discussions, reach out via **GitHub Issues**.

---
🚀 *Enjoy using the fine-tuned Bhagavad Gita LLM!* 🙏
