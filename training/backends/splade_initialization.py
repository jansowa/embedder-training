"""SPLADE decoder initialization and optimizer grouping; no eager ML imports."""
from __future__ import annotations

import json
import math
from contextlib import nullcontext
from pathlib import Path


def head_parameter_ids(model):
    auto = model[0].auto_model
    ids = {id(p) for name, p in auto.named_parameters(remove_duplicate=False)
           if name.startswith(('head.', 'decoder.'))}
    if not ids:
        raise ValueError('SPLADE head.* / decoder.* parameters not found')
    return ids


def configure_trainable(model, config):
    value = config.get('splade_train_only_head', False)
    if not isinstance(value, bool):
        raise ValueError('splade_train_only_head must be a boolean')
    if not value:
        return
    ids = head_parameter_ids(model)
    for parameter in model.parameters():
        parameter.requires_grad_(id(parameter) in ids)
    print('[INIT] Head-only training; tied decoder/input embedding weights remain shared.', flush=True)


def configure_head_optimizer(trainer, model, config):
    if 'splade_head_learning_rate' not in config:
        return  # the trainer already uses learning_rate for all parameters
    lr = float(config['splade_head_learning_rate'])
    if not math.isfinite(lr) or lr <= 0:
        raise ValueError('splade_head_learning_rate must be finite and positive')
    ids = head_parameter_ids(model)
    original = trainer.create_optimizer

    def create_optimizer():
        optimizer = original()
        optimizer = trainer.optimizer if optimizer is None else optimizer
        if getattr(optimizer, '_splade_head_groups', False):
            return optimizer
        groups = []
        for group in optimizer.param_groups:
            for head in (False, True):
                params = [p for p in group['params'] if (id(p) in ids) == head]
                if params:
                    new = dict(group, params=params)
                    if head:
                        new['lr'] = lr
                        new['initial_lr'] = lr
                    groups.append(new)
        optimizer.param_groups.clear()
        for group in groups:
            optimizer.add_param_group(group)
        optimizer._splade_head_groups = True
        return optimizer

    trainer.create_optimizer = create_optimizer


def quantized_counts(maxima, c=0.0):
    import torch
    return (torch.round(torch.log1p(torch.relu(maxima - c)) * 100) > 0).sum(-1)


def choose_bias_shift(maxima, target):
    import torch
    if target <= 0 or target >= maxima.shape[1]:
        raise ValueError('splade_decoder_bias_target must be between zero and vocabulary size')
    lo, hi = float(maxima.min()) - 1, float(maxima.max()) + 1
    for _ in range(32):
        mid = (lo + hi) / 2
        count = float(quantized_counts(maxima, mid).float().median())
        if count > target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def raw_maxima(model, texts, *, bf16=False, batch_size=4):
    import torch
    device = next(model.parameters()).device
    auto = model[0].auto_model
    values = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            features = model.tokenize(texts[start:start + batch_size])
            features = {k: v.to(device) for k, v in features.items() if k in ('input_ids', 'attention_mask', 'token_type_ids')}
            context = torch.autocast('cuda', dtype=torch.bfloat16) if bf16 and device.type == 'cuda' else nullcontext()
            with context:
                logits = auto(**features).logits
                # Max-before-log1p is mathematically the same stock SPLADE pooling.
                logits = logits.masked_fill(~features['attention_mask'].bool().unsqueeze(-1), -torch.inf)
                values.append(logits.amax(1).float().cpu())
    return torch.cat(values)


def microbatch(rows):
    selected = [r for r in rows if r.get('anchor') and r.get('positive')
                and all(r.get(f'negative_{i}') for i in range(1, 7))][:8]
    if len(selected) != 8:
        raise ValueError('Initialization diagnostics require 8 rows with six negatives')
    queries = [r['anchor'] for r in selected]
    docs = [r['positive'] for r in selected]
    for i in range(1, 7):
        docs.extend(r[f'negative_{i}'] for r in selected)
    return queries, docs


def score_diagnostics(model, rows, *, bf16=False):
    import torch
    queries, docs = microbatch(rows)
    q = raw_maxima(model, queries, bf16=bf16).relu().log1p()
    d = raw_maxima(model, docs, bf16=bf16).relu().log1p()
    scores = q @ d.T
    counts = quantized_counts(raw_maxima(model, docs, bf16=bf16))
    return dict(score_min=float(scores.min()), score_median=float(scores.median()),
                score_max=float(scores.max()), loss=float(torch.nn.functional.cross_entropy(scores, torch.arange(8))),
                document_active_dims=counts.tolist(), document_dims_median=float(counts.float().median()),
                document_dims_mean=float(counts.float().mean()),
                micro_batch=8, documents=56, scale=1.0, mode='eval', forward_bf16=bf16,
                diagnostic_scores_dtype='float32')


def initialize_decoder(model, rows, config, output_dir):
    raw = config.get('splade_decoder_bias_shift')
    if raw is None:
        return None
    import torch
    auto = model[0].auto_model
    bias = auto.get_output_embeddings().bias
    if bias is None:
        raise ValueError('SPLADE output decoder has no bias')
    if raw != 'auto' and (isinstance(raw, bool) or not math.isfinite(float(raw))):
        raise ValueError('splade_decoder_bias_shift must be auto or a finite float')
    was_training = model.training
    if torch.cuda.is_available():
        model.to('cuda')
    model.eval()
    original = bias.detach().clone()
    target = int(config.get('splade_decoder_bias_target', 400))
    if target <= 0:
        raise ValueError('splade_decoder_bias_target must be positive')
    bf16 = bool(config.get('bf16', False))
    try:
        docs = [r['positive'] for r in rows if r.get('positive')][:256]
        if len(docs) < 256:
            raise ValueError('Bias calibration requires 256 training passages')
        maxima = raw_maxima(model, docs, bf16=bf16)
        before = float(quantized_counts(maxima).float().median())
        attempts = []
        for current_target in ([target, 200] if raw == 'auto' and target != 200 else [target]):
            c = choose_bias_shift(maxima, current_target) if raw == 'auto' else float(raw)
            with torch.no_grad():
                bias.copy_(original - c)
            after = float(quantized_counts(raw_maxima(model, docs, bf16=bf16)).float().median())
            stats = score_diagnostics(model, rows, bf16=bf16)
            attempts.append(dict(c=c, target=current_target, median_dims_before=before, median_dims_after=after, **stats))
            print('[INIT] ' + json.dumps(attempts[-1]), flush=True)
            if stats['loss'] <= 15 or raw != 'auto':
                break
        result = dict(**attempts[-1], attempts=attempts, calibration_documents=len(docs))
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        (Path(output_dir) / 'splade_initialization.json').write_text(json.dumps(result, indent=2) + '\n')
        return result
    except Exception:
        with torch.no_grad():
            bias.copy_(original)
        raise
    finally:
        model.train(was_training)


def update_density_streak(previous, value):
    return previous + 1 if value is not None and float(value) < 5 else 0
