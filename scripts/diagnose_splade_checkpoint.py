"""Check SPLADE weights and raw outputs independently of retrieval evaluation."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', action='append', required=True, help='LABEL=MODEL')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--trainer-state', type=Path)
    args = parser.parse_args()
    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    assert torch.cuda.is_available(), 'Run through a queue on the idle GPU'
    texts = [
        'Warszawa jest stolicą Polski i leży nad Wisłą.',
        'Jakie miasto jest stolicą Polski?',
        'Podatek dochodowy jest naliczany od uzyskanego dochodu.',
        'Woda wrze w temperaturze stu stopni Celsjusza przy normalnym ciśnieniu.',
    ]
    report = {'texts': texts, 'models': [], 'history': []}
    if args.trainer_state:
        state = json.loads(args.trainer_state.read_text())
        for entry in state.get('log_history', []):
            if entry.get('step', 0) <= 300 or entry.get('step') in [500, 625, 1000, 2500, 5000]:
                report['history'].append(entry)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for spec in args.model:
        label, model_path = spec.split('=', 1)
        for dtype_name, dtype in [('fp32', torch.float32), ('bf16', torch.bfloat16)]:
            started = time.time()
            print(f'Checking {label} / {dtype_name}', flush=True)
            tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
            model = AutoModelForMaskedLM.from_pretrained(model_path, torch_dtype=dtype, local_files_only=True).to('cuda').eval()
            bad_weights = []
            head_weights = {}
            for name, tensor in model.state_dict().items():
                if tensor.is_floating_point() and not torch.isfinite(tensor).all().item():
                    bad_weights.append(name)
                if name.startswith(('head.', 'decoder.')):
                    head_weights[name] = {'shape': list(tensor.shape), 'min': float(tensor.min()), 'max': float(tensor.max()), 'norm': float(tensor.float().norm())}
            item = {'label': label, 'model': model_path, 'dtype': dtype_name,
                    'nonfinite_weights': bad_weights, 'head_weights': head_weights, 'outputs': []}
            for max_length in [256, 512]:
                encoded = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors='pt').to('cuda')
                torch.cuda.reset_peak_memory_stats()
                with torch.inference_mode():
                    logits = model(**encoded).logits
                    pooled = (torch.log1p(torch.relu(logits.float())) * encoded['attention_mask'].unsqueeze(-1)).amax(dim=1)
                    finite = torch.isfinite(pooled)
                    quantized = torch.round(pooled * 100)
                    metrics = {
                        'max_length': max_length, 'logits_all_finite': bool(torch.isfinite(logits).all()),
                        'logits_min': float(logits.min()), 'logits_max': float(logits.max()),
                        'pooled_all_finite': bool(finite.all()), 'pooled_max': float(pooled.max()),
                        'raw_active_dims_per_text': (pooled > 0).sum(dim=1).cpu().tolist(),
                        'quantized_active_dims_per_text': (quantized > 0).sum(dim=1).cpu().tolist(),
                        'valid_token_active_dims_per_text': (quantized[:, :len(tokenizer)] > 0).sum(dim=1).cpu().tolist(),
                        'peak_vram_gib': torch.cuda.max_memory_allocated() / 2**30,
                    }
                item['outputs'].append(metrics)
                del encoded, logits, pooled, finite, quantized
            item['seconds'] = time.time() - started
            report['models'].append(item)
            args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
            print(json.dumps({'label': label, 'dtype': dtype_name, 'bad_weights': bad_weights, 'outputs': item['outputs']}, ensure_ascii=False), flush=True)
            del model, tokenizer
            gc.collect()
            torch.cuda.empty_cache()
    print(f'Report: {args.output}', flush=True)


if __name__ == '__main__':
    main()
