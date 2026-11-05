"""Write a technical verdict for bounded ModernBERT collapse controls."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path("experiments/polish-splade-2026-10/results")


def main() -> None:
    diagnostic_path = ROOT / "data/modernbert-controls-diagnosis.json"
    diagnostics = json.loads(diagnostic_path.read_text()) if diagnostic_path.exists() else {"models": []}
    rows = []
    for name in ("noreg-scale1", "noreg-scale1e-4"):
        run_root = Path(f"runs/modernbert-control-{name}")
        telemetry_paths = list(run_root.rglob("collapse_control.jsonl"))
        telemetry = []
        for path in telemetry_paths:
            for line in path.read_text().splitlines():
                try:
                    telemetry.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        step_logs = [r for r in telemetry if "grad_norm" in r.get("metrics", {})]
        measured = [m for m in diagnostics["models"] if m["label"] == name]
        counts = {m["dtype"]: m["outputs"][0]["valid_token_active_dims_per_text"] for m in measured}
        if not measured:
            verdict = "INCOMPLETE: no completed direct model measurement"
        elif any(m["nonfinite_weights"] for m in measured):
            verdict = "FAIL: nonfinite model weights"
        elif any(not o["pooled_all_finite"] for m in measured for o in m["outputs"]):
            verdict = "FAIL: nonfinite pooled representations"
        elif all(all(v == 0 for v in o["valid_token_active_dims_per_text"]) for m in measured for o in m["outputs"]):
            verdict = "COLLAPSED: zero vectors in direct FP32/BF16 checks"
        else:
            verdict = "NONZERO on four diagnostic texts; retrieval quality remains unmeasured"
        row = {"name": name, "max_step": max((r["step"] for r in telemetry), default=None),
               "last_training_metrics": step_logs[-1]["metrics"] if step_logs else None,
               "counts": counts, "verdict": verdict,
               "stop_reasons": [r["stop_reason"] for r in telemetry if r.get("stop_reason")],
               "final_saved": bool(list(run_root.rglob("final/model.safetensors"))),
               "telemetry": telemetry}
        state_paths = list(run_root.rglob("control_train_state.json"))
        if state_paths:
            row["training_state"] = json.loads(state_paths[-1].read_text())
        rows.append(row)
    written = datetime.now(ZoneInfo("Europe/Warsaw")).isoformat(timespec="seconds")
    result = {"written": written, "controls": rows}
    (ROOT / "data/modernbert-controls-summary.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = ["# ModernBERT: bounded collapse controls", "", f"Written: {written}", "",
             "Both controls start from the original MLM, with zero query/document FLOPS weights, "
             "the same five training files, seed, physical batch 8 and accumulation 4. "
             "The 500-step cap preserves the original 250-step learning-rate warmup (ratio 0.5). "
             "Only ranking scale differs between the two controls: 1 versus 0.0001. "
             "The original failed run used nonzero, scheduled regularization. "
             "These controls do not test a complete corrected regularized recipe.", "",
             "| Control | Last step | FP32 active dimensions (four texts) | BF16 active dimensions | Verdict |",
             "|---|---:|---|---|---|"]
    for row in rows:
        lines.append(f"| {row['name']} | {row['max_step']} | {row['counts'].get('fp32', 'missing')} | "
                     f"{row['counts'].get('bf16', 'missing')} | {row['verdict']} |")
    first, soft = rows
    lines += ["", "## Interpretation", ""]
    if first["verdict"].startswith("COLLAPSED"):
        lines.append("Collapse is reproduced without FLOPS regularization: regularization is not necessary for "
                     "this failure under the original ranking scale and learning rate.")
    elif first["verdict"].startswith("NONZERO"):
        lines.append("Removing FLOPS avoids an all-zero final model on the diagnostic sample. "
                     "This implicates the original regularized recipe, but does not isolate individual "
                     "learning-rate/regularizer interactions or establish retrieval quality.")
    else:
        lines.append("The original-scale control did not produce a usable completed measurement; "
                     "inspect logs before drawing a causal conclusion.")
    if first["verdict"].startswith("COLLAPSED") and soft["verdict"].startswith("NONZERO"):
        lines.append("Reducing ranking scale while holding the no-regularization control fixed prevents "
                     "an all-zero result on the four texts. This supports score scale as a contributing "
                     "factor; retrieval evaluation and a regularized control are still required.")
    elif soft["verdict"].startswith("COLLAPSED"):
        lines.append("A softer ranking scale alone does not prevent collapse in this control. "
                     "Backbone learning rate or a frozen-backbone head warmup is a next controlled test.")
    lines += ["", "No full retraining, night-3 preparation, commit or push is scheduled by these diagnostic queues. "
              "All-zero and incomplete measurements are not reported as nDCG scores. "
              "Inspect `logs/mlm_control_*.log`, `logs/mlm_controls_diagnosis.log` and "
              "`logs/queue_5070_mlm_controls_finish.out` for errors. Raw results are under `results/data/`."]
    output = ROOT / "modernbert_controls_report.md"
    output.write_text("\n".join(lines) + "\n")
    print(output)


if __name__ == "__main__":
    main()
