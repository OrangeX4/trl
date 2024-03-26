# 0. imports
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import requests
from PIL import Image

from trl import AutoModelForVison2SeqLMWithValueHead, PPOConfig, PPOTrainer


# 1. load a pretrained model
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = AutoModelForVison2SeqLMWithValueHead.from_pretrained("microsoft/trocr-base-handwritten")
model_ref = AutoModelForVison2SeqLMWithValueHead.from_pretrained("microsoft/trocr-base-handwritten")

# 2. initialize trainer
ppo_config = {"mini_batch_size": 1, "batch_size": 1}
config = PPOConfig(**ppo_config)
ppo_trainer = PPOTrainer(config, model, model_ref, processor.tokenizer)

# 3. encode a query
# load image from the IAM dataset
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
query_tensor = processor(image, return_tensors="pt").pixel_values
print(query_tensor.shape)

# 4. generate model response
generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": processor.tokenizer.eos_token_id,
    "max_new_tokens": 20,
}
response_tensor = ppo_trainer.generate(list(query_tensor), return_prompt=False, **generation_kwargs)
response_txt = processor.batch_decode(response_tensor, skip_special_tokens=True)[0]

print('response:', response_txt)

# 5. define a reward for response
# (this could be any reward such as human feedback or output from another model)
reward = [torch.tensor(1.0, device=model.pretrained_model.device)]

# 6. train model with ppo
train_stats = ppo_trainer.step([query_tensor[0]], [response_tensor[0]], reward)
for key in train_stats:
    try:
        print(key, train_stats[key].shape)
    except AttributeError:
        print(key, train_stats[key])
print('----------------------------------------------------------------------')
train_stats = ppo_trainer.step([query_tensor[0]], [response_tensor[0]], reward)
for key in train_stats:
    try:
        print(key, train_stats[key].shape)
    except AttributeError:
        print(key, train_stats[key])

