# ShotContinuity-2B

ShotContinuity checks two adjacent shots in a storyboard or Blender production log. Supply tracked wardrobe, props or set details for both shots, plus a director's note. It reports which changes look unintended and what earlier value to restore.

This is a LoRA adapter for [Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B). It consumes text annotations. It does not inspect footage, open .blend files or edit a scene.

## Example

```json
{
  "scene": "Mountain observatory",
  "actor": "Kavi",
  "shot_a": {
    "dome": "closed",
    "chair": "inside",
    "journal": "shut",
    "red light": "dim"
  },
  "shot_b": {
    "red light": "bright",
    "dome": "open",
    "journal": "shut",
    "chair": "inside"
  },
  "director_note": "The note about the dome being open referred to an alternate edit we aren't using. Camera reframes a little closer in B. Continuity has signed off on the red light being bright from B onward."
}
```

```json
{
  "issues": [
    {
      "field": "dome",
      "restore": "closed"
    }
  ]
}
```

## Evaluation

| Final holdout, 24 synthetic shot pairs | Base | Adapter |
|---|---:|---:|
| Exact list of fixes | 2/24 | 18/24 |
| Responses matching the JSON schema | 10/24 | 24/24 |
| Correctly left fully authorized changes alone | 0/4 | 4/4 |
| One authorized change plus one mistake | 1/4 | 1/4 |
| One authorized change plus two mistakes | 0/4 | 1/4 |

The final set uses four fictional settings and director-note templates held apart from training and development. Both runs use the same prompt, greedy decoding and base checkpoint revision. The scorer accepts a JSON code fence and requires an exact field list and restore values. No production footage was used. A clean report still needs human review, and intentional changes can be flagged.

Training used 240 examples per epoch for 2 epochs and peaked at 5.47 GiB of CUDA allocations on a 12 GB RTX 4070 SUPER. The data is original and generated from a seeded Python script.

## Run the checker

The [GitHub project](https://github.com/Ikteder/ShotContinuity-2B) contains the CLI, training script, generated fixtures and evaluation outputs. The base model is loaded separately. A CUDA-enabled PyTorch install is needed for GPU use.

```shell
pip install -r requirements.txt
python check_shots.py example.json
```

## Reproduce

Place the official Qwen3.5-2B snapshot at revision `15852e8c16360a2fea060d615a32b45270f8a8fc` in a local directory. Create separate output directories for training and final evaluation.

```shell
python train_shot_continuity.py --root ./development --base-path ./qwen-base
python train_shot_continuity.py --root ./final --base-path ./qwen-base --evaluate-adapter ./development/adapter
```

The [Hugging Face release](https://huggingface.co/udoy55/ShotContinuity-2B) provides the PEFT adapter weights. The published evaluation files are synthetic. Review proposed fixes against your own shot notes before applying them.
