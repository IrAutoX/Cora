# Cora

Cora is a Persian-first AI training project. GitHub Actions trains a small language model for about two minutes, saves a checkpoint, reports metrics, packages the model, and creates a release.

## Training

```bash
python -m pip install -r requirements.txt
python train.py --minutes 2
python chat.py --checkpoint checkpoints/cora-latest.pt
```

The repository contains an original Persian seed corpus in `data/fa.txt`. Extend it only with data you have permission to use.

## Tools

Cora has an explicit local tool registry. Tools are only available when the user enables them with `--allow-tools`.