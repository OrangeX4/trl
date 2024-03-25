# 0. imports
import torch
from transformers import AutoTokenizer

from trl import AutoModelForSeq2SeqLMWithValueHead, PPOConfig, PPOTrainer


# 1. load a pretrained model
model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained("t5-small")
model_ref = AutoModelForSeq2SeqLMWithValueHead.from_pretrained("t5-small")
tokenizer = AutoTokenizer.from_pretrained("t5-small")
tokenizer.pad_token = tokenizer.eos_token

# 2. initialize trainer
ppo_config = {"mini_batch_size": 1, "batch_size": 1}
config = PPOConfig(**ppo_config)
ppo_trainer = PPOTrainer(config, model, model_ref, tokenizer)

# 3. encode a query
query_txt = "This morning I went to the "
query_tensor = tokenizer.encode(query_txt, return_tensors="pt").to(model.pretrained_model.device)

# 4. generate model response
generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
    "max_new_tokens": 20,
}
response_tensor = ppo_trainer.generate(list(query_tensor), return_prompt=False, **generation_kwargs)
response_txt = tokenizer.decode(response_tensor[0])

print('response:', response_txt)

# 5. define a reward for response
# (this could be any reward such as human feedback or output from another model)
reward = [torch.tensor(1.0, device=model.pretrained_model.device)]

# 6. train model with ppo
train_stats = ppo_trainer.step([query_tensor[0]], [response_tensor[0]], reward)
# print(train_stats)