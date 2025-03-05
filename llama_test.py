from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "mistralai/Mistral-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
llama3 = AutoModelForCausalLM.from_pretrained(model_name)

print(llama3)