import time
import requests
import json
import base64

url = 'http://localhost:8080/generate_stream'
headers = {'Content-Type': 'application/json'}

uri = "/data01/tuwenming/airbus_detect/airbus/val/images/d8873734-016a-4b9d-9b9e-8bc47eb13fef_1984_2048.jpg" # or "/http/path/of/image"
if uri.startswith("http"):
    images = [{"type": "url", "data": uri}]
else:
    with open(uri, 'rb') as fin:
        b64 = base64.b64encode(fin.read()).decode("utf-8")
    images=[{'type': "base64", "data": b64}]

data = {
    "req_id":"10",
    "prompt_len":"1000",
    "output_len":"20",
    "req_time": 30,
    "model_dir": "/data01/tuwenming/S-LoRA/models/LLava/llava-v1.5-7b",
    "lora_dir": "/data02/tuwenming/LLaVA/checkpoints/llava-v1.5-7b-task-lora-0",
    "inputs": "<image>Generate the caption in English with grounding:",
    "parameters": {
        "max_new_tokens": 30,
        # The space before <|endoftext|> is important, the server will remove the first bos_token_id, but QWen tokenizer does not has bos_token_id
    },
    "multimodal_params": {
        "images": images,
    }
}
# print(data)

response = requests.post(url, headers=headers, data=json.dumps(data))
if response.status_code == 200:
    print(response.text)
else:
    print('Error:', response.status_code, response.text)