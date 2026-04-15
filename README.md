# FixMatch, BaseLine(Supervised)

## Command-Line Arguments

| Argument | Type | Description |
|---------|------|-------------|
| `--mode` | str | Operation to run (`pretrain`, `supervised`, `linear`, `inference`) |
| `--config` | str | Path to YAML config file |
| `--checkpoint-dir` | str | Directory to save checkpoints |
| `--device` | str | Device to train on (`cpu` or `cuda`) |
| `--label-pct` | float | Portion of labeled data for supervised training (e.g., 0.1, 0.25, 0.5, 1.0) |
