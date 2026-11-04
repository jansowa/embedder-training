# Resume the paused MLM probe on the 5070

Training was paused on 3 November 2026 at 08:50 CET, after a complete save at
step **3750 / 18750** (120,000 examples). The optimizer, scheduler, RNG state,
trainer state and model weights were verified before the queue and its child
processes were stopped. The GPU had no remaining compute processes.

After turning the computer back on, run in **WSL**:

```bash
cd /mnt/c/work/embedder-training-v2
bash experiments/polish-splade-2026-10/queues/resume_5070_mlm1.sh
```

This uses `mlm_probe_a_resume.yaml`, the original output directory and its latest
complete `checkpoint-*` (3750 at the time of the pause). It restores the full
training state; `max_steps` stays **18750 total**, not 18750 additional steps.
Physical batch 8, accumulation 4, gradient checkpointing and pooling chunk 8
remain unchanged, as do the data, learning rate and regularizer weights.

The wrapper refuses to start if the checkpoint or optimizer state is missing.
It also verifies the resolved configuration against the immutable run manifest.
The resume config is copied from the saved resolved config, changing only resume
flags. The physical free-space gate is 25 GiB for a resume (35 GiB for a fresh
probe): the saved run already occupies 5.6 GiB, leaving about 7-8 GiB additional
model/checkpoint storage plus benchmark cache and reserve.
The existing queue skips completed anchors and later measures the preserved
probe checkpoints. Do not use the original fresh-start launcher for this resume.

The queue deadline defaults to 18 hours after invocation, so it cannot expire
while the computer is off. One training invocation still has a 12-hour timeout;
if necessary, another invocation of the same resume wrapper continues from the
latest complete checkpoint. No training is launched by the pause preparation.

Keep `runs/arm-mlm-probe-a` and its checkpoints, `.venv`, datasets and model cache.
The checkpoint path and stop record are in `logs/mlm_paused.json`.

The timeout was extended on 3 November via a verified step-11250 checkpoint; the run continues to 18750 total steps.
