import evaluate
import numpy as np
import os

import torch
from pathlib import Path
from typing import Dict
from transformers import EvalPrediction, RobertaTokenizer, AutoImageProcessor, AutoModel
from torch.nn.functional import cosine_similarity

from PIL import Image
import typst
import io

MITEX_FILE_PATH = Path(__file__).resolve().parent / "mitex.typ"
with open(MITEX_FILE_PATH, "w") as f:
    f.write("")
compiler = typst.Compiler(MITEX_FILE_PATH)


def mitex(latex, ppi=144.0):
    template = f"""
    #import "@preview/mitex:0.2.3": *
    #set page(height: auto, width: auto, margin: 0em)
    #mitex(`
    {latex}
    `)
    """
    with open(MITEX_FILE_PATH, "w") as f:
        f.write(template)
    res = compiler.compile(format="png", ppi=ppi)
    return res


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
model = AutoModel.from_pretrained("google/vit-base-patch16-224").to(DEVICE)


def bleu_metric(eval_preds: EvalPrediction, tokenizer: RobertaTokenizer) -> Dict:
    cur_dir = Path(os.getcwd())
    os.chdir(Path(__file__).resolve().parent)
    metric = evaluate.load(
        "google_bleu"
    )  # Will download the metric from huggingface if not already downloaded
    os.chdir(cur_dir)

    logits, labels = eval_preds.predictions, eval_preds.label_ids
    preds = logits

    labels = np.where(labels == -100, 1, labels)

    preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    return metric.compute(predictions=preds, references=labels)


def image_similarity_score(pred_images, gt_images):
    pred_inputs = processor(pred_images, return_tensors="pt").to(DEVICE)
    pred_outputs = model(**pred_inputs)
    gt_inputs = processor(gt_images, return_tensors="pt").to(DEVICE)
    gt_outputs = model(**gt_inputs)
    similarity_score = cosine_similarity(
        pred_outputs.pooler_output, gt_outputs.pooler_output, dim=1
    )
    return similarity_score


def formula_similarity_score(pred_formulas, gt_formulas, fail_score=0.0):
    pred_images = []
    gt_images = []
    success = []
    for formula in pred_formulas:
        try:
            img = Image.open(io.BytesIO(mitex(formula))).convert("RGB")
            success.append(True)
        except Exception as e:
            img = Image.new("RGB", (224, 224))
            success.append(False)
        pred_images.append(img)
    gt_images = [
        Image.open(io.BytesIO(mitex(formula))).convert("RGB") for formula in gt_formulas
    ]
    scores = image_similarity_score(pred_images, gt_images)
    for i, s in enumerate(success):
        if not s:
            scores[i] = fail_score
    return scores


if __name__ == "__main__":
    pred_formulas = [
        r"\int_{-\infty}^{\infty} e^{-x^2} dx = \sqrt{\pi}",
        r"\int_{-\infty}^{\infty} e^{-x^2} dx = \sqrt{x}",
        r"\int_{-\infty}^{\infty} e^{-x^2} dx = \sqrfdsf",
        r"\int_{-\infty}^{\infty} e^{-x^2} dx",
    ]
    gt_formulas = [
        r"\int_{-\infty}^{\infty} e^{-x^2} dx = \sqrt{\pi}",
        r"\int_{-\infty}^{\infty} e^{-x^2} dx = \sqrt{\pi}",
        r"\int_{-\infty}^{\infty} e^{-x^2} dx = \sqrt{\pi}",
        r"\int_{-\infty}^{\infty} e^{-x^2} dx = \sqrt{\pi}",
    ]
    print(formula_similarity_score(pred_formulas, gt_formulas))
