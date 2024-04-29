import sys

sys.path.append("/home/orangex4/projects/trl")
import os
from datetime import datetime
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
from ..utils.metrics import formula_similarity_score
from ...globals import MAX_TOKEN_SIZE, MIN_WIDTH, MIN_HEIGHT


def build_dataset():
    script_dirpath = Path(__file__).resolve().parent
    os.chdir(script_dirpath)

    dataset = load_dataset(str(Path("./dataset/loader.py").resolve()))["train"]
    dataset = dataset.filter(
        lambda x: x["image"].height > MIN_HEIGHT and x["image"].width > MIN_WIDTH
    )
    dataset = dataset.shuffle(seed=42)
    dataset = dataset.flatten_indices()
    dataset = dataset.map(
        img_process_fn, batched=True, remove_columns=dataset.column_names
    )
    return dataset


def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])


def generate_dir_name():
    # seed0-03-29-16-16-4150244
    return f"seed{CONFIG['seed']}-{datetime.now().strftime('%m-%d-%H-%M-%S-%f')[:-3]}-{os.getpid()}"


if __name__ == "__main__":
    # set global seed
    torch.manual_seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])
    dir_name = generate_dir_name()
    config = PPOConfig(
        **CONFIG["ppo_config"],
        accelerator_kwargs={"project_dir": "logs/default/" + dir_name},
        seed=CONFIG["seed"],
    )
    dataset = build_dataset()
    model = AutoModelForVison2SeqLMWithValueHead.from_pretrained(
        CONFIG["pretrained_model"]
    )
    ref_model = AutoModelForVison2SeqLMWithValueHead.from_pretrained(
        CONFIG["pretrained_model"]
    )
    tokenizer = TexTeller.get_tokenizer(CONFIG["pretrained_model"])
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
        "pad_token_id": tokenizer.eos_token_id,
    }
    for iteration, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        query_tensors = [
            torch.from_numpy(np.array(pv)).to(ppo_trainer.accelerator.device)
            for pv in batch["pixel_values"]
        ]

        #### Get response
        response_tensors = []
        for query in query_tensors:
            response = ppo_trainer.generate([query], **generate_config)
            response_tensors.append(response[0])
        batch["response"] = [
            tokenizer.decode(r.squeeze(), skip_special_tokens=True)
            for r in response_tensors
        ]

        #### Compute similarity score
        rewards = list(formula_similarity_score(batch["response"], batch["formula"]))

        #### Run PPO step
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        ppo_trainer.log_stats(stats, batch, rewards)

        # save checkpoints
        if iteration % CONFIG["save_steps"] == 0:
            model.save_pretrained(
                f"train_result/default/{dir_name}/checkpoint-{iteration}"
            )

        with open(f"logs/default/{dir_name}/trl/records.txt", "a") as f:
            for i, reward in enumerate(rewards):
                f.write(f"{batch['formula'][i]}\n")
                f.write(f"{batch['response'][i]}\n")
                f.write(f"{reward}\n")
