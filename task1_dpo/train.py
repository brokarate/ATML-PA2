from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from common.data import (
    encode_prompt_response,
    load_yaml,
    pad_batch,
    preference_responses,
    prompt_messages_from_preference,
    read_jsonl,
    repo_path,
)
from common.generation import response_sequence_logprobs
from common.logging_utils import append_jsonl, save_json, set_seed, wall_timer
from common.models import load_policy, load_tokenizer, reference_mode, trainable_parameters
from task1_dpo.dpo import dpo_loss


def make_collate(tokenizer, max_length):
    def collate(rows):
        chosen, rejected = [], []
        for row in rows:
            prompt = prompt_messages_from_preference(row)
            yc, yr = preference_responses(row)
            try:
                chosen.append(encode_prompt_response(tokenizer, prompt, yc, max_length))
                rejected.append(encode_prompt_response(tokenizer, prompt, yr, max_length))
            except ValueError as exc:
                if "Prompt alone has" not in str(exc):
                    raise
                continue
        if not chosen:
            raise ValueError(
                "All examples in this batch had prompts longer than max_sequence_length. "
                "Increase max_sequence_length or use a smaller batch after filtering."
            )
        return pad_batch(tokenizer, chosen), pad_batch(tokenizer, rejected)
    return collate


def prompt_token_count(tokenizer, row: dict) -> int:
    prompt_ids = tokenizer.apply_chat_template(
        prompt_messages_from_preference(row),
        tokenize=True,
        add_generation_prompt=True,
    )
    if hasattr(prompt_ids, "input_ids"):
        prompt_ids = prompt_ids.input_ids
    if prompt_ids and isinstance(prompt_ids[0], list):
        prompt_ids = prompt_ids[0]
    return len(prompt_ids)


def filter_overlong_prompts(tokenizer, rows: list[dict], max_length: int) -> tuple[list[dict], list[dict]]:
    kept, skipped = [], []
    for idx, row in enumerate(rows):
        tokens = prompt_token_count(tokenizer, row)
        if tokens >= max_length:
            skipped.append({"index": idx, "prompt_tokens": tokens})
        else:
            kept.append(row)
    return kept, skipped


def prepare_dpo_run(config_path: str, dataset_path: str | None = None, beta: float | None = None, max_examples: int | None = None):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))
    path = dataset_path or cfg["paths"]["dpo_standard_train"]
    rows = read_jsonl(path)
    if max_examples is not None:
        rows = rows[: int(max_examples)]

    tokenizer = load_tokenizer(cfg["base_model"])
    rows, skipped = filter_overlong_prompts(tokenizer, rows, int(cfg["max_sequence_length"]))
    if skipped:
        print(f"Skipped overlong prompts: {len(skipped)}")
    model = load_policy(cfg, trainable=True, fresh_lora=True)
    loader = DataLoader(
        rows,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        collate_fn=make_collate(tokenizer, int(cfg["max_sequence_length"])),
    )
    optimizer = AdamW(
        trainable_parameters(model),
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
    )
    return {
        "cfg": cfg,
        "rows": rows,
        "skipped_overlong_prompts": skipped,
        "tokenizer": tokenizer,
        "model": model,
        "loader": loader,
        "optimizer": optimizer,
        "beta": float(cfg["beta"] if beta is None else beta),
    }


def run_training(config_path: str, run_name: str, dataset_path: str | None = None, output_path: str | None = None, beta: float | None = None, max_examples: int | None = None):
    bundle = prepare_dpo_run(config_path, dataset_path, beta, max_examples)
    cfg = bundle["cfg"]
    output = repo_path(output_path or cfg["standard_output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)

    model = bundle["model"]
    tokenizer = bundle["tokenizer"]
    optimizer = bundle["optimizer"]
    loader = bundle["loader"]
    beta_value = bundle["beta"]
    grad_accum_steps = int(cfg["grad_accum_steps"])
    max_grad_norm = float(cfg.get("max_grad_norm", 1.0))
    results_dir = repo_path(cfg["results_dir"]) / run_name
    train_log = results_dir / "train_log.jsonl"
    if train_log.exists():
        train_log.unlink()
    skipped_path = results_dir / "skipped_overlong_prompts.json"
    if bundle["skipped_overlong_prompts"]:
        save_json(skipped_path, bundle["skipped_overlong_prompts"])

    timer = wall_timer()
    optimizer.zero_grad(set_to_none=True)
    global_step = 0
    examples_seen = 0

    for epoch in range(int(cfg["epochs"])):
        progress = tqdm(loader, desc=f"{run_name} epoch {epoch + 1}/{cfg['epochs']}")
        for batch_idx, (chosen, rejected) in enumerate(progress, start=1):
            device = next(model.parameters()).device
            chosen = {k: v.to(device) for k, v in chosen.items()}
            rejected = {k: v.to(device) for k, v in rejected.items()}

            policy_chosen, _, _ = response_sequence_logprobs(model, chosen)
            policy_rejected, _, _ = response_sequence_logprobs(model, rejected)
            with torch.no_grad(), reference_mode(model):
                ref_chosen, _, _ = response_sequence_logprobs(model, chosen)
                ref_rejected, _, _ = response_sequence_logprobs(model, rejected)

            loss, diagnostics = dpo_loss(
                policy_chosen,
                policy_rejected,
                ref_chosen,
                ref_rejected,
                beta_value,
            )
            (loss / grad_accum_steps).backward()
            examples_seen += int(chosen["input_ids"].shape[0])
            progress.set_postfix(
                loss=float(loss.detach().cpu()),
                acc=float(diagnostics["preference_accuracy"].cpu()),
            )

            should_step = batch_idx % grad_accum_steps == 0 or batch_idx == len(loader)
            if not should_step:
                continue

            torch.nn.utils.clip_grad_norm_(trainable_parameters(model), max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            append_jsonl(
                train_log,
                {
                    "run": run_name,
                    "epoch": epoch + 1,
                    "step": global_step,
                    "examples_seen": examples_seen,
                    "beta": beta_value,
                    "loss": float(loss.detach().cpu()),
                    **{
                        key: float(value.detach().cpu())
                        for key, value in diagnostics.items()
                    },
                },
            )

    model.save_pretrained(output)
    tokenizer.save_pretrained(output)
    summary = {
        "run": run_name,
        "beta": beta_value,
        "train_file": str(dataset_path or cfg["paths"]["dpo_standard_train"]),
        "train_examples": len(bundle["rows"]),
        "skipped_overlong_prompts": len(bundle["skipped_overlong_prompts"]),
        "output_dir": str(output.relative_to(repo_path(".")) if output.is_relative_to(repo_path(".")) else output),
        "optimizer_steps": global_step,
        "wall_time_seconds": timer(),
    }
    save_json(results_dir / "summary.json", summary)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--run-name", default="standard")
    ap.add_argument("--dataset")
    ap.add_argument("--output")
    ap.add_argument("--beta", type=float)
    ap.add_argument("--max-examples", type=int)
    args = ap.parse_args()
    run_training(args.config, args.run_name, args.dataset, args.output, args.beta, args.max_examples)


if __name__ == "__main__":
    main()
