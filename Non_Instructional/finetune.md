# Non‑Instructional Fine‑Tuning with QLoRA

This project demonstrates how to fine‑tune a small language model using **QLoRA (4‑bit quantization + LoRA adapters)** on a raw text dataset.

The notebook uses:

* **TinyLlama-1.1B-Chat-v1.0**
* **4‑bit quantization (NF4)** for memory efficiency
* **LoRA adapters** for parameter‑efficient fine‑tuning
* A **non‑instructional dataset** (plain text instead of prompt/response pairs)

## What This Notebook Covers

* Loading a quantized language model using Hugging Face Transformers
* Configuring QLoRA with BitsAndBytes
* Preparing raw text datasets for causal language modeling
* Tokenization and fixed‑length chunking
* Training with the Hugging Face Trainer API
* Using LoRA for efficient fine‑tuning on limited GPU memory

## Tech Stack

* Python
* PyTorch
* Hugging Face Transformers
* PEFT (LoRA)
* BitsAndBytes
* Datasets

## Dataset

The notebook uses Shakespeare text data as an example of non‑instructional training.

## Goal

The main objective is to understand how modern LLM fine‑tuning works while keeping GPU memory usage low enough for consumer GPUs or Kaggle environments.

## Run the Notebook

Install dependencies:

```bash
pip install -U transformers accelerate bitsandbytes peft datasets
```

Then run the notebook cells step by step.

## Learning Outcomes

By completing this notebook, you will understand:

* Basics of QLoRA
* 4‑bit model loading
* LoRA adapter training
* Causal language modeling
* Non‑instructional fine‑tuning workflows
