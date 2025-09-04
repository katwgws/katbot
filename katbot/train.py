import os
import random
import warnings
from datetime import datetime
from typing import Any, cast

import numpy as np
import torch
from datasets import Dataset, load_dataset
from dotenv import load_dotenv
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer  # type: ignore

# silence some minor warning spam
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
warnings.filterwarnings("ignore", category=UserWarning, module="peft")

if not torch.cuda.is_available():
    raise RuntimeError("CUDA not found, exiting")
USE_BF16 = torch.cuda.is_bf16_supported()
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")

load_dotenv()


# - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # -


_seed = os.getenv("SEED")
if _seed:
    SEED = int(_seed)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
else:
    SEED = None

DRY_RUN = os.getenv("DRY_RUN", "").casefold().strip() in {"true", "yes", "1"}
DRY_SAMPLE = 512

BASE_MODEL = os.getenv("BASE_MODEL", "")
if not BASE_MODEL:
    raise RuntimeError("BASE_MODEL not specified")
KATBOT_MODEL = os.getenv("KATBOT_MODEL", "")
if not KATBOT_MODEL:
    raise RuntimeError("KATBOT_MODEL not specified")
if DRY_RUN:
    KATBOT_MODEL += "-DRY"

MAX_SEQ_LEN = 128
EVAL_SPLIT = 0.08
RECENCY_ALPHA = 0.2
TWEET_TOKEN = "<|tweet|>"

DATA_DIR = "./data"
IN_PATH = DATA_DIR + "/training_corpus.jsonl"
MODEL_DIR = DATA_DIR + "/" + KATBOT_MODEL
ADAPTER_DIR = MODEL_DIR + "/adapter"
MERGED_DIR = MODEL_DIR + "/merged"

BNB_CFG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16 if USE_BF16 else torch.float16,
)

LORA_CFG = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    modules_to_save=["embed_tokens", "lm_head"],
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
)

SFT_CFG = SFTConfig(
    output_dir=MODEL_DIR,
    max_length=MAX_SEQ_LEN,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-4,
    warmup_steps=128,
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="epoch",
    save_total_limit=1,
    bf16=USE_BF16,
    fp16=not USE_BF16,
    gradient_checkpointing=True,
    dataloader_num_workers=4,
    report_to=["none"],
    dataset_text_field="text",
)

if DRY_RUN:
    SFT_CFG.gradient_accumulation_steps = 4
    SFT_CFG.learning_rate = 8e-4
    SFT_CFG.warmup_steps = 10
    SFT_CFG.eval_strategy = "no"
    SFT_CFG.save_strategy = "no"

TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
TOP_P = float(os.getenv("TOP_P", "0.9"))
REPETITION_PENALTY = float(os.getenv("REPETITION_PENALTY", "1.08"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "32"))


# - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # -


def _load_dataset(in_path: str, eval_split: float) -> tuple[Dataset, Dataset]:
    print(f"Loading dataset from {in_path} ...")
    ds = load_dataset("json", data_files={"train": in_path}, split="train")
    split = cast(Dataset, ds).train_test_split(test_size=eval_split, seed=SEED)
    train_ds, eval_ds = split["train"], split["test"]

    if DRY_RUN:
        train_ds = train_ds.select(range(min(len(train_ds), DRY_SAMPLE)))
        eval_ds = eval_ds.select(range(min(len(eval_ds), DRY_SAMPLE // 4)))

    return train_ds, eval_ds


def _reweight_dataset(ds: Dataset, alpha: float) -> Dataset:
    print("Applying recency bias ...")
    ts = np.array(
        [datetime.fromisoformat(str(d)).timestamp() for d in ds["date"]],
        dtype=np.float64,
    )
    if ts.min() == ts.max():
        return ds  # all weights equal, prevents divide by 0

    score = (ts - ts.min()) / (ts.max() - ts.min())
    w = 1.0 + alpha * (np.exp(score) - 1) / (np.e - 1)
    idx = np.random.choice(len(ds), size=len(ds), replace=True, p=(w / w.sum()))

    return ds.select(idx.tolist()).shuffle(SEED)


def _map_dataset(train_ds: Dataset, eval_ds: Dataset, eos_token: str) -> tuple[Dataset, Dataset]:
    print("Mapping to text columns ...")

    def _to_text(e: dict[str, Any]) -> dict[str, Any]:
        return {"text": TWEET_TOKEN + e["text"].strip() + eos_token}

    return (
        train_ds.map(_to_text, remove_columns=train_ds.column_names),
        eval_ds.map(_to_text, remove_columns=eval_ds.column_names),
    )


def _make_tokenizer(model_name: str, max_len: int):
    print("Loading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.model_max_length = max_len
    if TWEET_TOKEN not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": [TWEET_TOKEN]})
    return tokenizer


def _make_base_model(model_name: str, tokenizer_size: int):
    print("Loading base model ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=BNB_CFG, device_map="auto"
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    model.resize_token_embeddings(tokenizer_size)
    return model


# - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # -


def train(
    *,
    dataset_path: str = IN_PATH,
    eval_split: float = EVAL_SPLIT,
    recency_alpha: float = RECENCY_ALPHA,
    base_model_name: str = BASE_MODEL,
    max_sequence_length: int = MAX_SEQ_LEN,
    lora_cfg=LORA_CFG,
    sft_cfg=SFT_CFG,
    output_dir: str = ADAPTER_DIR,
) -> None:
    print("Initializing ...")
    if SEED:
        print(f"Using fixed seed '{SEED}'.")
    if DRY_RUN:
        print(
            f"Dry run enabled. Sample size limited to {DRY_SAMPLE:,} "
            + f"for training and {DRY_SAMPLE // 4:,} for eval."
        )

    train_ds, eval_ds = _load_dataset(dataset_path, eval_split)

    if not DRY_RUN:
        train_ds = _reweight_dataset(train_ds, recency_alpha)

    tokenizer = _make_tokenizer(base_model_name, max_sequence_length)
    train_ds, eval_ds = _map_dataset(train_ds, eval_ds, tokenizer.eos_token)

    model = _make_base_model(base_model_name, len(tokenizer))
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        peft_config=lora_cfg,
        args=SFTConfig(
            **sft_cfg.to_dict()
            | {"eos_token": tokenizer.eos_token}
            | {"pad_token": tokenizer.pad_token}
            | ({"seed": SEED} if SEED else {})
        ),
        processing_class=tokenizer,
    )

    print("Setup complete! Training ...")
    trainer.train()

    print("Saving model ...")
    trainer.model.save_pretrained(  # type: ignore
        output_dir, safe_serialization=True, save_embedding_layers=True
    )

    print("Saving tokenizer ...")
    tokenizer.save_pretrained(output_dir)

    print("Training complete!")


# - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # -


def merge(
    *,
    base_model_name: str = BASE_MODEL,
    adapter_dir: str = ADAPTER_DIR,
    output_dir: str = MERGED_DIR,
) -> None:
    print("Merging ...")
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir, use_fast=True)
    base = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map="cpu",
        dtype=torch.float16,
    )
    base.resize_token_embeddings(len(tokenizer))

    peft_model = PeftModel.from_pretrained(base, adapter_dir).eval()
    merged = peft_model.merge_and_unload()  # type: ignore

    merged.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)


# - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # -


def sample(
    num: int = 6,
    *,
    temperature: float = TEMPERATURE,
    top_p: float = TOP_P,
    repetition_penalty: float = REPETITION_PENALTY,
    max_new_tokens: int = MAX_TOKENS,
    model_dir: str = MERGED_DIR,
) -> None:
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        device_map="auto",
        dtype=torch.bfloat16 if USE_BF16 else torch.float16,
    ).eval()

    inputs = tokenizer(
        [TWEET_TOKEN] * num,
        add_special_tokens=False,
        return_tensors="pt",
        padding=True,
    ).to(model.device)
    length = inputs.input_ids.shape[1]

    outputs = model.generate(
        **inputs,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    for i, out in enumerate(outputs):
        tweet = tokenizer.decode(out[length:], skip_special_tokens=True).strip()
        print(f"[{i + 1}] {tweet}")


# - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # -


def main():
    import multiprocessing as mp

    mp.freeze_support()  # stop recursive imports on win32

    train()
    merge()
    sample()


if __name__ == "__main__":
    main()
