"""Fit a LoRA adapter for shot continuity and score it against the pinned base."""
import argparse
from collections import Counter
import json
from pathlib import Path
import random
import time

from shot_continuity import SYSTEM, compact, generate

REVISION = "15852e8c16360a2fea060d615a32b45270f8a8fc"
BASE = "Qwen/Qwen3.5-2B"


def validate(rows):
    for row in rows:
        shot = row["input"]
        expected = row["expected"]["issues"]
        assert len({entry["field"] for entry in expected}) == len(expected)
        for entry in expected:
            assert shot["shot_a"][entry["field"]] == entry["restore"]
            assert shot["shot_b"][entry["field"]] != entry["restore"]
        assert (row["mode"] in ("clean", "authorized")) == (not expected)


def parse(text):
    value = text.strip()
    if value.startswith("```") and value.endswith("```"):
        value = value.split("\n", 1)[1][:-3].strip()
    data = json.loads(value)
    if not isinstance(data, dict) or set(data) != {"issues"} or not isinstance(data["issues"], list):
        raise ValueError("Expected a JSON object with an issues array")
    if any(not isinstance(item, dict) or set(item) != {"field", "restore"}
           for item in data["issues"]):
        raise ValueError("Unexpected issue schema")
    return data


def run(args):
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    start = time.monotonic()
    torch.manual_seed(2701)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    root = args.root
    base_path = args.base_path
    train = list(generate("train"))
    holdout = list(generate("final" if args.evaluate_adapter else "eval"))
    validate(train + holdout)
    assert {r["scene"] for r in train}.isdisjoint(r["scene"] for r in holdout)
    for filename, data in (("train.jsonl", train), ("holdout.jsonl", holdout)):
        (root / filename).write_text("".join(compact(row) + "\n" for row in data), encoding="utf8")
    model = AutoModelForImageTextToText.from_pretrained(base_path, dtype=torch.bfloat16,
                                                       device_map={"": "cuda"}, local_files_only=True)
    model.config.use_cache = False
    tokenizer = AutoTokenizer.from_pretrained(base_path, local_files_only=True)
    trainable = model
    result = {}
    summary = {}

    def encode(row):
        return tokenizer.apply_chat_template([
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": compact(row["input"])}],
            tokenize=True, add_generation_prompt=True, enable_thinking=False, return_dict=False)

    def evaluate(tag):
        trainable.eval()
        items = []
        score_by_mode = Counter()
        for index, row in enumerate(holdout, start=1):
            ids = torch.tensor([encode(row)], device="cuda")
            with torch.inference_mode():
                answer = trainable.generate(ids, attention_mask=torch.ones_like(ids),
                                            do_sample=False, max_new_tokens=96, use_cache=True)
            text = tokenizer.decode(answer[0, ids.shape[1]:], skip_special_tokens=True).strip()
            try:
                parsed = parse(text)
                valid = True
                exact = parsed == row["expected"]
            except (ValueError, IndexError):
                valid, exact = False, False
            items.append({"scene": row["scene"], "mode": row["mode"], "output": text,
                          "expected": row["expected"], "json_valid": valid, "exact": exact})
            if exact:
                score_by_mode[row["mode"]] += 1
            if index % 12 == 0:
                print(tag, index, "/", len(holdout), "exact so far", sum(i["exact"] for i in items), flush=True)
        result[tag] = items
        summary[tag] = {"exact": sum(i["exact"] for i in items), "total": len(items),
                        "json_valid": sum(i["json_valid"] for i in items), "by_mode": dict(score_by_mode)}
        (root / "evaluation.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf8")
        (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf8")
        print(tag, summary[tag], flush=True)

    evaluate("base")
    if args.evaluate_adapter:
        trainable = PeftModel.from_pretrained(model, args.evaluate_adapter)
        evaluate("adapter")
        return

    trainable = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, task_type="CAUSAL_LM",
                                               target_modules=["q_proj", "v_proj", "in_proj_qkv"]))
    trainable.print_trainable_parameters()
    trainable.train()
    optimizer = torch.optim.AdamW([p for p in trainable.parameters() if p.requires_grad], lr=2e-4)
    sampler = random.Random(2701)
    for epoch in range(args.epochs):
        sampler.shuffle(train)
        batch = train[:args.examples_per_epoch]
        optimizer.zero_grad(set_to_none=True)
        window_loss = 0
        for index, row in enumerate(batch, start=1):
            prompt = encode(row)
            reply = tokenizer.encode(compact(row["expected"]) + "<|im_end|>", add_special_tokens=False)
            ids = torch.tensor([prompt + reply], device="cuda")
            logits = trainable(input_ids=ids, use_cache=False, logits_to_keep=len(reply) + 1).logits
            loss = torch.nn.functional.cross_entropy(logits[0, :-1].float(), ids[0, -len(reply):])
            (loss / 4).backward()
            window_loss += loss.item()
            if index % 4 == 0:
                torch.nn.utils.clip_grad_norm_(trainable.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            if index % 24 == 0:
                print(f"epoch {epoch + 1}: {index}/{len(batch)} loss {window_loss / 24:.3f}", flush=True)
                window_loss = 0
    adapter = root / "adapter"
    trainable.save_pretrained(adapter)
    config_path = adapter / "adapter_config.json"
    config = json.loads(config_path.read_text())
    config.update(base_model_name_or_path=BASE, revision=REVISION)
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf8")
    evaluate("adapter")
    summary["train"] = {"revision": REVISION, "training_pool": len(train), "examples_per_epoch": args.examples_per_epoch,
                        "epochs": args.epochs, "peak_cuda_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                        "wall_seconds": round(time.monotonic() - start, 1)}
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf8")
    print("training and evaluation complete", summary, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--base-path", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--examples-per-epoch", type=int, choices=(120, 240), default=240)
    parser.add_argument("--evaluate-adapter", type=Path, help="Score a saved adapter on the final holdout")
    run(parser.parse_args())
