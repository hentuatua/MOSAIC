# MOSAIC

MOSAIC is a two-stage model for diversity-aware image retrieval. The query module adapts image representations with respect to the query. The set scorer then ranks candidate images with set-level coverage and diversity objectives.

This repository contains code only. Datasets, checkpoints, logs, and generated results should be kept outside the source tree.

## Structure

- `mosaic/query_module/`: query-conditioned representation modules and stage-1 losses.
- `mosaic/set_score/`: set-aware scoring model, diverse decoding, losses, and evaluation.
- `mosaic/data/`: dataset loading and metric utilities.
- `mosaic/train.py`: training entry point.
- `mosaic/evaluate.py`: evaluation entry point.

## Data Layout

Pass split directories with `--train_dataset` and `--test_dataset`. Each split is expected to use the following layout:

```text
<split>/
  data.json
  img/
    <query_name>/
      <image_id>.jpg
  feats/
    <feature_extractor>/
      img/
        <query_name>/
          <image_id>.pt
      txt/
        <query_text>.pt
  gt/
    dGT/
      <query_name> dGT.txt
      <query_name> dclusterGT.txt
    rGT/
      <query_name> rGT.txt
```

`data.json` should contain one record per candidate image, including the query label, subtopic label, relevance label, and feature path fields used by the dataset reader. Existing diversity retrieval datasets such as Div400 and Div150Cred can be converted to this layout.

Feature tensors are expected to be precomputed before training. The default feature extractor name is `clip`, but any extractor can be used if its feature dimension is registered in the code.

## Training

From the repository root:

```bash
python -m mosaic.train \
  --train_dataset /path/to/devset \
  --test_dataset /path/to/testset \
  --save_dir /path/to/run_dir
```

By default, stage 1 runs for 5 epochs and writes `stage1_best.pt`. Stage 2 uses that checkpoint and writes `budget_best.pt` under `--save_dir`.

For a quick syntax and data-loading check, use a small query limit:

```bash
python -m mosaic.train \
  --train_dataset /path/to/devset \
  --test_dataset /path/to/testset \
  --save_dir /path/to/run_dir \
  --dry_run \
  --max_train_queries 2 \
  --max_eval_queries 2
```

## Evaluation

```bash
python -m mosaic.evaluate \
  --checkpoint /path/to/budget_best.pt \
  --stage1_ckpt /path/to/stage1_best.pt \
  --test_dataset /path/to/testset
```

Use `--json_out /path/to/metrics.json` to save the metric summary.

## Environment

Install PyTorch and NumPy versions that match your CUDA or CPU environment.
