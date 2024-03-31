import sys

sys.path.append('/home/orangex4/projects/trl')
import os
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from trl import (
    AutoModelForVison2SeqLMWithValueHead,
    PPOTrainer,
    PPOConfig,
)

from .training_args import CONFIG
from ..model.TexTeller import TexTeller
from ..utils.functional import img_process_fn
from ...globals import MAX_TOKEN_SIZE, MIN_WIDTH, MIN_HEIGHT


def build_dataset():
    script_dirpath = Path(__file__).resolve().parent
    os.chdir(script_dirpath)

    dataset = load_dataset(str(Path('./dataset/loader.py').resolve()))['train']
    dataset = dataset.filter(
        lambda x: x['image'].height > MIN_HEIGHT and x['image'].width > MIN_WIDTH
    )
    dataset = dataset.shuffle(seed=42)
    dataset = dataset.flatten_indices()
    dataset = dataset.map(
        img_process_fn, batched=True, remove_columns=dataset.column_names, num_proc=8
    )
    return dataset


def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])


if __name__ == '__main__':
    config = PPOConfig(**CONFIG['ppo_config'])
    dataset = build_dataset()
    model = AutoModelForVison2SeqLMWithValueHead.from_pretrained(
        CONFIG['pretrained_model']
    )
    ref_model = AutoModelForVison2SeqLMWithValueHead.from_pretrained(
        CONFIG['pretrained_model']
    )
    tokenizer = TexTeller.get_tokenizer(CONFIG['pretrained_model'])
    ppo_trainer = PPOTrainer(config, model, ref_model, tokenizer)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=ppo_trainer.config.batch_size,
        collate_fn=collator,
        shuffle=True,
        drop_last=True,
    )
    generate_config = {
        **CONFIG["generate_config"],
        "max_new_tokens": MAX_TOKEN_SIZE,
        "pad_token_id": tokenizer.eos_token_id,
    }
    for epoch, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        query_tensors = [
            torch.from_numpy(np.array(pv)).to(ppo_trainer.accelerator.device)
            for pv in batch["pixel_values"]
        ]

        #### Get response
        response_tensors = []
        for query in query_tensors:
            response = ppo_trainer.generate([query], **generate_config)
            response_tensors.append(response[0])
        batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]

        #### Compute sentiment score
        # texts = [q + r for q, r in zip(batch["query"], batch["response"])]
        # pipe_outputs = sentiment_pipe(texts, **sent_kwargs)
        rewards = [
            torch.tensor(1.0, device=model.pretrained_model.device)
            for output in batch["response"]
        ]

        #### Run PPO step
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        ppo_trainer.log_stats(stats, batch, rewards)
        print(stats)
