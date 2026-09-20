# Cora AI

Cora is a Persian-first coding assistant built on Qwen2.5-0.5B-Instruct and fine-tuned with LoRA on Persian conversations and real Python instruction/code data.

## What changed

- No character-level echo model.
- Qwen2.5-0.5B-Instruct base model.
- Persian conversational training.
- Python programming instruction/code training.
- One-hour GitHub Actions training by default.
- Checkpoint at 2 minutes, then periodic checkpoints.
- Final LoRA adapter and tokenizer are released.
- Training metrics are printed and stored in `checkpoints/metrics.json`.
- Web access, Python execution, system info and package installation tools are available when explicitly enabled.
- Multi-turn chat memory.
- Private reasoning is not printed; Cora produces only the useful answer and plan.

## Local run

```bash
python -m pip install -r requirements.txt
python train.py --minutes 60
python chat.py --checkpoint checkpoints/cora-latest --allow-tools
```

## Tools

```text
/tools
/tool web_answer Python decorators چیست؟
/tool web_get https://docs.python.org/3/
/tool python print(2 + 2)
/tool install requests
```

Tool execution is explicit and user-controlled.

## Datasets

Training streams from `jtatman/python-code-dataset-500k` for Python and `saied/Persian_Chat_Dataset` for Persian, plus the local `data/fa.txt` seed corpus. The Python dataset is MIT-labelled but its card warns that its GitHub-derived source material may have varied upstream licensing, so production/commercial redistribution should be reviewed separately.

## GitHub Actions

Pushes to `main` and manual workflow dispatch start training. The default manual run is 60 minutes. After training, Actions prints metrics, creates a compressed model artifact, calculates SHA-256, uploads artifacts and creates a GitHub Release.

The base model is Qwen2.5-0.5B-Instruct, an Apache-2.0 model with multilingual and coding capabilities.
