"""Read two annotated shots and print proposed continuity fixes as JSON."""
import argparse
import json
from pathlib import Path
import sys

from shot_continuity import SYSTEM, compact
from train_shot_continuity import BASE, REVISION, parse


def validate_request(request):
    if not isinstance(request, dict) or not isinstance(request.get("director_note"), str):
        raise ValueError("Input must contain a director_note string")
    a, b = request.get("shot_a"), request.get("shot_b")
    if not isinstance(a, dict) or not isinstance(b, dict) or not a or set(a) != set(b):
        raise ValueError("shot_a and shot_b must be nonempty objects with matching field names")
    if any(not isinstance(v, str) for v in (*a.values(), *b.values())):
        raise ValueError("Tracked values must be strings")


def infer(request, adapter, device, base_path=None):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    validate_request(request)
    base = base_path or BASE
    tokenizer = AutoTokenizer.from_pretrained(base, revision=REVISION)
    model = AutoModelForImageTextToText.from_pretrained(
        base, revision=REVISION, dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map={"": device})
    model = PeftModel.from_pretrained(model, adapter).eval()
    ids = tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": compact(request)}],
        tokenize=True, add_generation_prompt=True, enable_thinking=False, return_dict=False)
    ids = torch.tensor([ids], device=device)
    with torch.inference_mode():
        answer = model.generate(ids, attention_mask=torch.ones_like(ids), do_sample=False, max_new_tokens=128)
    result = parse(tokenizer.decode(answer[0, ids.shape[1]:], skip_special_tokens=True))
    seen = set()
    for issue in result["issues"]:
        field = issue["field"]
        if (not isinstance(field, str) or field in seen or field not in request["shot_a"]
                or issue["restore"] != request["shot_a"][field]
                or issue["restore"] == request["shot_b"][field]):
            raise ValueError("The model returned a fix inconsistent with the supplied shots")
        seen.add(field)
    return result


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("input", type=Path)
    cli.add_argument("--adapter", default="udoy55/ShotContinuity-2B")
    cli.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    cli.add_argument("--base-path", type=Path, help="Optional local folder for the pinned Qwen checkpoint")
    args = cli.parse_args()
    try:
        shot = json.loads(args.input.read_text(encoding="utf-8"))
        print(json.dumps(infer(shot, args.adapter, args.device, args.base_path), indent=2, ensure_ascii=False))
    except ValueError as error:
        print(error, file=sys.stderr)
        sys.exit(1)
