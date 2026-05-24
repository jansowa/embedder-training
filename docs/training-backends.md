# Training backends

The main training entry point is:

```bash
python -m training.train --backend flagembedding --training-type embedder --config configs/grid.yaml
```

CLI values take precedence over `backend` and `training_type` values from the YAML config. If neither the CLI nor the config provides them, the CLI defaults to `flagembedding + embedder` for compatibility with the previous pipeline.

## Installation

This repository does not currently use `pyproject.toml`, so backend-specific environments are split into requirements files:

```bash
pip install -r requirements/requirements-sentence-transformers.txt
pip install -r requirements/requirements-flagembedding.txt
pip install -r requirements/requirements-pylate.txt
pip install -r requirements/requirements-all-backends.txt
```

The root `requirements.in` remains the full-environment variant for compatibility with the existing README and lock workflow.

## Backends and Training Types

| Backend | Training type | Status |
| --- | --- | --- |
| `flagembedding` | `embedder` | Implemented. Runs the existing pipeline via `torchrun -m FlagEmbedding.finetune.embedder.encoder_only.base`. |
| `sentence-transformers` | `embedder` | Implemented. Trains a dense embedder on JSONL `query`/`pos`/`neg` records with `MultipleNegativesRankingLoss` and offline negatives. |
| `sentence-transformers` | `matryoshka` | Implemented. Dense training with `MatryoshkaLoss` wrapping `MultipleNegativesRankingLoss`. |
| `sentence-transformers` | `splade` | Implemented. Sparse training with `SparseEncoder`, `SparseMultipleNegativesRankingLoss`, and `SpladeLoss`. |
| `sentence-transformers` | `multimodal` | Extension point, raises `NotImplementedError`. |
| `sentence-transformers` | `adaptive-layer` | Extension point, raises `NotImplementedError`. |
| `sentence-transformers` | `matryoshka-2d` | Extension point, raises `NotImplementedError`. |
| `pylate` | `colbert` | Implemented. Trains ColBERT with PyLate `ColBERT`, `Contrastive`, and `ColBERTCollator`. |
| `pylate` | `late-interaction` | Implemented as an alias for the same PyLate ColBERT loop. |

Unsupported combinations are rejected before the backend module is loaded, for example:

```text
Training type 'splade' is not supported by backend 'pylate'. Supported types: colbert, late-interaction
```

## Example Commands

```bash
python -m training.train --backend flagembedding --training-type embedder --config configs/grid.yaml
python -m training.train --backend sentence-transformers --training-type embedder --config configs/smoke_sentence_transformers_embedder.yaml
python -m training.train --backend sentence-transformers --training-type embedder --config configs/smoke_sentence_transformers_filtered_embedder.yaml
python -m training.train --backend sentence-transformers --training-type matryoshka --config configs/smoke_sentence_transformers_matryoshka.yaml
python -m training.train --backend sentence-transformers --training-type splade --config configs/smoke_sentence_transformers_splade.yaml
python -m training.train --backend pylate --training-type colbert --config configs/smoke_pylate_colbert.yaml
```

The legacy wrapper still works:

```bash
python run_experiments.py --run_pirb --remove_checkpoints --pirb_scope small
```

In `train` mode it delegates to:

```bash
python -m training.train --backend flagembedding --training-type embedder --config configs/grid.yaml
```

## Configuration

All backends support the same grid shape. The CLI expands `architectures` x
`hparams` before it calls the selected backend. Each expanded architecture is
exposed to the backend as `model_name_or_path`, and each hparams mapping is
applied as per-run overrides.

```yaml
backend: sentence-transformers
training_type: splade
runs_dir: runs/polish-splade
train_data: ./dataset-small-no_in_batch_neg

architectures:
  - sdadas/polish-distilroberta
  - sdadas/polish-roberta-base-v2
hparams:
  - learning_rate: 2e-6
    num_train_epochs: 1

sentence_transformers:
  train_batch_size: 1
  negatives_per_query: 1
```

The existing `configs/grid.yaml` FlagEmbedding format is still supported:

```yaml
backend: flagembedding
training_type: embedder
runs_dir: runs
wandb_project: mining-tests

architectures:
  - TaylorAI/bge-micro-v2
hparams:
  - learning_rate: 8e-5
    num_train_epochs: 1

backend_config:
  train_group_size: 6

flagembedding:
  sentence_pooling_method: mean

sentence_transformers: {}
pylate: {}
```

If a grid run does not set `output_dir`, SentenceTransformers and PyLate write
each expanded run under `runs_dir/<backend>/<training_type>/<run-name>`.

CLI values such as `--backend pylate --training-type colbert` override `backend` and `training_type` from YAML.

## CLI Parameters

| Parameter | Required | Description |
| --- | --- | --- |
| `--backend` | No | Training backend. Supported values: `flagembedding`, `sentence-transformers`, `pylate`. If omitted, the CLI reads `backend` from YAML and then defaults to `flagembedding`. |
| `--training-type` | No | Training recipe for the selected backend. The CLI validates the `backend + training_type` combination before loading the backend module. |
| `--config` | No | Path to the YAML config file. Defaults to `configs/grid.yaml`. |
| `--run-mteb` / `--run_mteb` | No | Runs MTEB after FlagEmbedding training. Leave disabled for smoke tests. |
| `--run-pirb` / `--run_pirb` | No | Runs PIRB after FlagEmbedding training. Leave disabled for smoke tests. |
| `--benchmark-name` | No | MTEB benchmark name. Defaults to `NanoBEIR`. Used only with `--run-mteb`. |
| `--pirb-scope` / `--pirb_scope` | No | PIRB scope: `tiny`, `small`, or `all`. Used only with `--run-pirb`. |
| `--remove-checkpoints` / `--remove_checkpoints` | No | Removes `checkpoint-*` directories after a successful FlagEmbedding run. |

The CLI only overrides `backend` and `training_type`. All other training parameters are read from YAML.

## YAML Parameters

### Shared Fields

| Field | Backends | Description |
| --- | --- | --- |
| `backend` | All | Default backend if `--backend` is not provided. |
| `training_type` | All | Default training type if `--training-type` is not provided. |
| `architectures` | All | Optional grid list. Each value becomes `model_name_or_path` for one expanded run. |
| `hparams` | All | Optional grid list of per-run overrides. Combined with every architecture. |
| `runs_dir` | All | Base directory for grid output directories when `output_dir` is omitted. |
| `train_data` | All | Dataset path. For SentenceTransformers and PyLate this can point to a directory containing `dataset.jsonl` or directly to a JSONL file. |
| `output_dir` | SentenceTransformers, PyLate | Training output directory. The final model is saved under `output_dir/final`. |
| `model_name_or_path` | All | Hugging Face model name or local model path. Grid `architectures` values are expanded into this field. |
| `max_steps` | SentenceTransformers, PyLate, FlagEmbedding | Maximum number of training steps. Smoke configs use `1`. |
| `num_train_epochs` | All | Number of epochs when `max_steps` does not stop training earlier. |
| `train_batch_size` / `per_device_train_batch_size` | All | Per-device training batch size. |
| `learning_rate` | All | Learning rate. |
| `weight_decay` | SentenceTransformers, PyLate, FlagEmbedding | Weight decay. |
| `warmup_ratio` | All | Warmup ratio. |
| `gradient_accumulation_steps` | All | Number of gradient accumulation steps. |
| `logging_steps` | All | Logging frequency. |
| `save_strategy`, `save_steps`, `save_total_limit` | All | Checkpoint behavior. Smoke configs use `save_strategy: "no"` so only the final model is saved. |
| `seed` | SentenceTransformers, PyLate | Seed passed to training arguments. |
| `fp16`, `bf16` | SentenceTransformers, PyLate, FlagEmbedding | Mixed precision settings. |
| `dataloader_drop_last`, `dataloader_num_workers` | All | Dataloader settings. |
| `report_to` | SentenceTransformers, PyLate | Reporting integrations, for example `[]` for smoke tests without W&B. |
| `backend_config` | All | Shared backend override section. Backend-specific sections such as `sentence_transformers` take precedence over `backend_config`. |
| `dataset_filter` | FlagEmbedding, SentenceTransformers | Optional YAML filter profile. The filtered dataset is materialized before training. |
| `dataset_filter_cache_dir` | FlagEmbedding, SentenceTransformers | Optional cache root for materialized filtered datasets. Defaults to `cache/filtered_datasets`. |

### FlagEmbedding

`flagembedding + embedder` keeps the existing grid workflow. Key fields:

| Field | Description |
| --- | --- |
| `architectures` | List of base models to iterate over, for example `TaylorAI/bge-micro-v2`. |
| `hparams` | List of hyperparameter variants. Each entry is combined with each model from `architectures`. |
| `runs_dir` | Directory for training runs. Defaults to `runs`. |
| `wandb_project` | W&B project. Defaults to `mining-tests` or `WANDB_PROJECT`. |
| `flagembedding.train_data` | Dataset in the format accepted by FlagEmbedding. |
| `flagembedding.train_group_size` | Number of items in each training group. |
| `flagembedding.query_max_len`, `flagembedding.passage_max_len` | Maximum query and passage lengths. |
| `flagembedding.query_instruction_for_retrieval` | Query instruction prefix. Defaults to `query: `. |
| `flagembedding.sentence_pooling_method` | Pooling strategy passed to FlagEmbedding. Defaults to `cls`. |
| `flagembedding.normalize_embeddings` | Whether to normalize embeddings. |
| `flagembedding.temperature` | Loss temperature. |
| `flagembedding.negatives_cross_device` | Whether to use cross-device negatives. |
| `flagembedding.deepspeed` | Path to a DeepSpeed config. |
| `flagembedding.gradient_checkpointing` | Whether to enable gradient checkpointing. |

Fields from `flagembedding` and `backend_config` are forwarded as arguments to `torchrun -m FlagEmbedding.finetune.embedder.encoder_only.base`, so an option supported by FlagEmbedding can usually be added to YAML without changing the CLI.

### SentenceTransformers

Shared parameters for `embedder`, `matryoshka`, and `splade`:

| Field | Description |
| --- | --- |
| `sentence_transformers.model_name_or_path` | Dense or sparse model. For SPLADE choose a masked-LM model, for example `hf-internal-testing/tiny-random-BertForMaskedLM` for smoke tests. |
| `sentence_transformers.train_data` | Overrides top-level `train_data`. |
| `sentence_transformers.output_dir` | Overrides top-level `output_dir`. |
| `sentence_transformers.negatives_per_query` | Number of offline negatives read from `neg`. If omitted, the backend uses the minimum negative count shared by all dataset rows. |
| `sentence_transformers.query_prefix` | Prefix added to `query` while loading the dataset. |
| `sentence_transformers.passage_prefix` | Prefix added to `pos` and `neg` while loading the dataset. |
| `sentence_transformers.max_seq_length` | Model maximum sequence length. |
| `sentence_transformers.model_kwargs` | Mapping passed to the model constructor. |
| `sentence_transformers.processor_kwargs` / `sentence_transformers.tokenizer_args` | Mapping passed to the processor/tokenizer loader for SPLADE models. |
| `sentence_transformers.tokenizer_name_or_path` | Optional tokenizer path for SPLADE models whose tokenizer lives in a separate checkpoint. |
| `sentence_transformers.run_name` | Optional run name passed to SentenceTransformers/Transformers. Grid runs generate unique names automatically. |
| `sentence_transformers.trust_remote_code` | Added to `model_kwargs` when set. |

Parameters only for `matryoshka`:

| Field | Description |
| --- | --- |
| `sentence_transformers.matryoshka_dims` | List of embedding dimensions, for example `[128, 64, 32]`. If omitted, the backend derives it from the model embedding dimension. |
| `sentence_transformers.matryoshka_weights` | Optional weights for Matryoshka dimensions. |
| `sentence_transformers.n_dims_per_step` | Number of dimensions sampled per step, passed to `MatryoshkaLoss`. |

Parameters only for `splade`:

| Field | Description |
| --- | --- |
| `sentence_transformers.document_regularizer_weight` | Document regularization weight for `SpladeLoss`. Defaults to `3e-5`. |
| `sentence_transformers.query_regularizer_weight` | Query regularization weight for `SpladeLoss`. Defaults to `5e-5`. |
| `sentence_transformers.scale` | Scale for `SparseMultipleNegativesRankingLoss`. Defaults to `1.0`. |
| `sentence_transformers.gather_across_devices` | Whether to gather embeddings across devices for a larger negative pool. |
| `sentence_transformers.splade_pooling_chunk_size` | Optional `SpladePooling(chunk_size=...)`. Smaller values reduce pooling memory usage at the cost of speed. |

### PyLate

Parameters for `colbert` and `late-interaction`:

| Field | Description |
| --- | --- |
| `pylate.model_name_or_path` | Base model for `pylate.models.ColBERT`. |
| `pylate.train_data` | Overrides top-level `train_data`. |
| `pylate.output_dir` | Overrides top-level `output_dir`. |
| `pylate.negatives_per_query` | Number of offline negatives read from `neg`. Smoke configs use `1`. |
| `pylate.query_prefix`, `pylate.passage_prefix` | Prefixes added while loading data. |
| `pylate.embedding_size` | ColBERT projection size. The smoke config uses `16`. |
| `pylate.query_length`, `pylate.document_length` | Query and document tokenization lengths. |
| `pylate.temperature` | Temperature for `pylate.losses.Contrastive`. Defaults to `1.0`. |
| `pylate.gather_across_devices` | Whether to gather representations across devices. |
| `pylate.model_kwargs` | Mapping passed to the base model. |
| `pylate.tokenizer_kwargs`, `pylate.config_kwargs` | Extra tokenizer and config arguments. |
| `pylate.trust_remote_code`, `pylate.revision`, `pylate.local_files_only`, `pylate.token` | Standard Hugging Face model loading settings. |
| `pylate.truncate_dim`, `pylate.bias`, `pylate.add_special_tokens`, `pylate.truncation` | Options forwarded to `pylate.models.ColBERT`. |
| `pylate.do_query_expansion`, `pylate.attend_to_expansion_tokens`, `pylate.skiplist_words` | ColBERT/PyLate-specific settings. |

## Dataset Format

SentenceTransformers and PyLate training use FlagEmbedding-style JSONL records:

```json
{"query": "...", "pos": ["..."], "neg": ["...", "..."]}
```

For dataset filtering, the recommended metadata layout keeps the training fields unchanged and stores per-item metadata in parallel lists:

```json
{
  "query": "co oznacza przegroda nosowa",
  "pos": ["positive passage 1", "positive passage 2"],
  "neg": ["negative passage 1", "negative passage 2"],
  "features": {
    "query": {
      "categories": {"language": "pl"},
      "flags": {"has_clean_query": true}
    },
    "pos": [
      {"ranks": {"cross_encoder_v1": 0.91}},
      {"ranks": {"cross_encoder_v1": 0.74}}
    ],
    "neg": [
      {"ranks": {"hard_negative_score": 0.72}},
      {"ranks": {"hard_negative_score": 0.35}}
    ]
  }
}
```

`features.pos[i]` describes `pos[i]`, and `features.neg[i]` describes `neg[i]`. If present, these metadata lists must have the same length as their passage lists. Legacy parallel fields such as `pos_scores`, `neg_scores`, `pos_id`, and `neg_id` are also kept aligned when filters remove passages.

Each positive passage creates a separate `anchor`/`positive` example. Offline negatives are forwarded as `negative_1`, `negative_2`, and so on. Dense SentenceTransformers training uses `MultipleNegativesRankingLoss`, so the model sees both in-batch negatives and offline negatives. `matryoshka` wraps the same loss with `MatryoshkaLoss`.

`sentence-transformers + splade` uses the same loader, but builds `MLMTransformer + SpladePooling` inside a
`SparseEncoder` and trains with `SpladeLoss`. This forces the fill-mask/MLM path that SPLADE needs.

`pylate + colbert` and `pylate + late-interaction` use the same JSONL format. The backend loads `pylate.models.ColBERT`, trains with `pylate.losses.Contrastive`, and batches with `pylate.utils.ColBERTCollator`.

## Dataset Filters

Encoder-only FlagEmbedding and SentenceTransformers runs can filter the training JSONL before the backend sees it:

```yaml
train_data: smoke-data/filtering-smoke
dataset_filter: configs/dataset_filters/example.yaml
dataset_filter_cache_dir: cache/filtered_datasets
```

For FlagEmbedding grid runs, `dataset_filter` and `dataset_filter_cache_dir` can be set globally, under `backend_config`, under `flagembedding`, or inside an individual `hparams` entry. The `hparams` value wins. For SentenceTransformers, the filter can be top-level, under `backend_config`, or under `sentence_transformers`.

The existing sample-level filter profile remains supported:

```yaml
name: pl_high_quality
version: 1
missing_policy: fail
type_mismatch_policy: fail

rules:
  - field: features.ranks.cross_encoder_v1
    op: gte
    value: 0.8
  - field: features.flags.is_synthetic
    op: eq
    value: false
  - field: features.categories.language
    op: in
    values: ["pl", "en"]
  - field: features.category_lists.domains
    op: intersects
    values: ["medical", "legal"]
```

Rules use dot-paths, so adding a new feature only requires a new path in YAML, for example `features.ranks.teacher_score` or `features.category_lists.retrievers`.

Passage-level filters use `sample_rules`, `positive_rules`, and `negative_rules`:

```yaml
name: passage_level_quality
version: 1
missing_policy: fail
type_mismatch_policy: fail
min_positives: 1
min_negatives: 1

sample_rules:
  - field: features.query.categories.language
    op: in
    values: ["pl", "en"]

positive_rules:
  - field: ranks.cross_encoder_v1
    op: gte
    value: 0.8

negative_rules:
  - field: ranks.hard_negative_score
    op: gte
    value: 0.6
```

`rules` is a backward-compatible alias for `sample_rules`. `positive_rules` are evaluated against one `features.pos[i]` object at a time, so their paths are relative to that object. `negative_rules` work the same way for `features.neg[i]`.

When a passage-level rule removes a positive or negative, the filter trims the corresponding text list and the aligned metadata lists. If fewer than `min_positives` or `min_negatives` remain, the full sample is removed.

See `configs/dataset_filters/example.yaml` for a sample-level profile and `configs/dataset_filters/passage_level_example.yaml` for a passage-level profile.

Supported operators:

| Kind | Operators |
| --- | --- |
| Numeric | `gt`, `gte`, `lt`, `lte`, `between` |
| Equality | `eq`, `neq` |
| Categorical | `in`, `not_in` |
| Lists | `intersects`, `contains_any`, `contains_all`, `contains_none` |
| Presence | `exists`, `missing` |

Logical groups are supported and evaluated strictly:

```yaml
rules:
  - any:
      - field: features.ranks.cross_encoder_v1
        op: gte
        value: 0.8
      - field: features.ranks.teacher_score
        op: gte
        value: 0.75
    missing_policy: exclude
```

Because the default `missing_policy` is `fail`, every missing field used by a profile fails the run, even inside `any`. Use `missing_policy: exclude` or `include` on a rule/group when missing optional features are expected. `type_mismatch_policy` follows the same `fail | include | exclude` choices.

Aggregates are available for list/map fields:

```yaml
rules:
  - field: features.ranks
    aggregate: max
    op: gte
    value: 0.85
```

Supported aggregates are `min`, `max`, `mean`, `sum`, and `count`.

When `train_data` is a directory, filtering picks `dataset.jsonl`, then `mixed_dataset.jsonl`, then the first `*.jsonl`, matching the existing loader behavior. The output is always a directory containing `dataset.jsonl` plus `filter_report.json`, under `cache/filtered_datasets/<profile>-<hash>/` by default. The hash includes the input path, input JSONL contents, and canonical filter profile, so different filters do not overwrite each other.

After filtering, the console reports the profile name, input path, output path, total/kept/removed counts, and missing/type mismatch counts per field. `filter_report.json` also includes positive/negative total, kept, and removed counts, plus samples removed by `min_positives` or `min_negatives`. On cache hits, the cached report is printed without re-filtering.

## Lazy Imports

`training.train` and the backend registry do not globally import `FlagEmbedding`, `sentence_transformers`, or `pylate`. Only the selected backend module is loaded, and the backend library is checked inside that module.

If the selected library is missing, the CLI shows a readable error instead of a raw `ModuleNotFoundError`, for example:

```text
Backend 'sentence-transformers' requires the sentence-transformers dependency. Install it with: pip install -r requirements/requirements-sentence-transformers.txt
```

FlagEmbedding checkpoint conversion to SentenceTransformers format runs only when `--run-mteb` or `--run-pirb` is enabled.

## Smoke Tests From Scratch

The following commands assume a fresh checkout and no prepared environment. They use `uv`, matching the main README. For another CUDA version, change `--extra-index-url` according to the README.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
exec "$SHELL"
```

The smoke tests use the mini dataset `dataset-small-no_in_batch_neg` and the small dense model `sentence-transformers-testing/stsb-bert-tiny-safetensors`. Benchmarks are intentionally disabled, so these commands do not run checkpoint conversion, MTEB, or PIRB.

### FlagEmbedding: Minimal Real Training

```bash
uv venv --python 3.10.4 .venv-flagembedding
source .venv-flagembedding/bin/activate

uv pip install -r requirements/requirements-flagembedding.txt \
  --extra-index-url https://download.pytorch.org/whl/cu126 \
  --index-url https://pypi.org/simple \
  --index-strategy unsafe-best-match

WANDB_MODE=disabled python -m training.train \
  --backend flagembedding \
  --training-type embedder \
  --config configs/smoke_flagembedding_embedder.yaml

deactivate
```

Expected result: the script runs `torchrun` for one training step (`max_steps: 1`) and writes output under `runs/smoke/`.

### SentenceTransformers: Minimal Dense Training

```bash
uv venv --python 3.10.4 .venv-sentence-transformers
source .venv-sentence-transformers/bin/activate

uv pip install -r requirements/requirements-sentence-transformers.txt \
  --extra-index-url https://download.pytorch.org/whl/cu126 \
  --index-url https://pypi.org/simple \
  --index-strategy unsafe-best-match

python -m training.train \
  --backend sentence-transformers \
  --training-type embedder \
  --config configs/smoke_sentence_transformers_embedder.yaml

deactivate
```

Expected result: the script runs one dense SentenceTransformers training step and saves the model to `runs/smoke/sentence-transformers/final`.

Filtered variant:

```bash
python -m training.train \
  --backend sentence-transformers \
  --training-type embedder \
  --config configs/smoke_sentence_transformers_filtered_embedder.yaml
```

Expected result: the script materializes a filtered dataset under `cache/filtered_datasets/`, runs one dense training step, and saves the model to `runs/smoke/sentence-transformers-filtered/final`.

### SentenceTransformers: Minimal Matryoshka Training

```bash
uv venv --python 3.10.4 .venv-sentence-transformers-matryoshka
source .venv-sentence-transformers-matryoshka/bin/activate

uv pip install -r requirements/requirements-sentence-transformers.txt \
  --extra-index-url https://download.pytorch.org/whl/cu126 \
  --index-url https://pypi.org/simple \
  --index-strategy unsafe-best-match

python -m training.train \
  --backend sentence-transformers \
  --training-type matryoshka \
  --config configs/smoke_sentence_transformers_matryoshka.yaml

deactivate
```

Expected result: the script runs one Matryoshka training step and saves the model to `runs/smoke/sentence-transformers-matryoshka/final`.

### SentenceTransformers: Minimal SPLADE Training

```bash
uv venv --python 3.10.4 .venv-sentence-transformers-splade
source .venv-sentence-transformers-splade/bin/activate

uv pip install -r requirements/requirements-sentence-transformers.txt \
  --extra-index-url https://download.pytorch.org/whl/cu126 \
  --index-url https://pypi.org/simple \
  --index-strategy unsafe-best-match

python -m training.train \
  --backend sentence-transformers \
  --training-type splade \
  --config configs/smoke_sentence_transformers_splade.yaml

python -m training.train \
  --backend sentence-transformers \
  --training-type splade \
  --config configs/smoke_sentence_transformers_splade_herbert.yaml

deactivate
```

Expected result: each script runs one SPLADE training step and saves the model under `runs/smoke/`.

### PyLate: Minimal ColBERT Training

```bash
uv venv --python 3.10.4 .venv-pylate-colbert
source .venv-pylate-colbert/bin/activate

uv pip install -r requirements/requirements-pylate.txt \
  --extra-index-url https://download.pytorch.org/whl/cu126 \
  --index-url https://pypi.org/simple \
  --index-strategy unsafe-best-match

python -m training.train \
  --backend pylate \
  --training-type colbert \
  --config configs/smoke_pylate_colbert.yaml

deactivate
```

Expected result: the script runs one ColBERT training step and saves the model to `runs/smoke/pylate-colbert/final`.

### Quick Error Checks

Unsupported combination:

```bash
python -m training.train \
  --backend pylate \
  --training-type splade \
  --config configs/smoke_pylate_colbert.yaml
```

Expected message:

```text
Training type 'splade' is not supported by backend 'pylate'. Supported types: colbert, late-interaction
```

Main CLI import without optional backends:

```bash
python -c "import training.train; print('training CLI import ok')"
```
