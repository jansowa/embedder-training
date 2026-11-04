"""Control of the SPLADE FLOPS regularizer weights during training.

Two mechanisms, both driven by ``sentence_transformers.regularizer_control``:

* a quadratic warm-up of both weights, ``lambda(t) = lambda_target * min(1, (t / T) ** 2)`` (SPLADE v2,
  ``naver/splade``), and
* a density controller: every ``interval_steps`` after the warm-up the callback measures the mean
  number of active vocabulary dimensions on a fixed sample of training documents, compares it with
  the density of a reference model on the same sample, and multiplies or divides the document weight
  by ``factor`` when the ratio leaves the dead band.

The module imports neither torch nor sentence-transformers at import time; both are imported inside
the functions that measure density, so ``training.train`` stays light.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import gc
import json
import math
import os
from pathlib import Path
from typing import Any, Callable, Mapping

from training.distributed import is_main_process


CONFIG_KEY = "regularizer_control"
CONFIG_PATH = f"sentence_transformers.{CONFIG_KEY}"
HISTORY_FILE_NAME = "regularizer_control.json"
DEFAULT_PREFIX = "regularizer"
# sentence-transformers >= 5.0 inserts this callback for every SpladeLoss; it ramps both weights from
# 0 to their configured values over the first third of the run and would fight the controller.
BUILTIN_WEIGHT_SCHEDULER_NAMES = frozenset({"SpladeRegularizerWeightSchedulerCallback"})

_TOP_LEVEL_KEYS = frozenset({"warmup_steps", "density"})
_DENSITY_KEYS = frozenset(
    {
        "start_step",
        "reference_model",
        "reference_dims",
        "band",
        "interval_steps",
        "factor",
        "min_document_weight",
        "max_document_weight",
        "query_ratio",
        "sample_size",
        "batch_size",
        "quantization_factor",
        "max_seq_length",
    }
)
_DENSITY_DEFAULTS: dict[str, Any] = {
    "band": (0.75, 1.05),
    "interval_steps": 250,
    "factor": 1.25,
    "min_document_weight": 5e-4,
    "max_document_weight": 3e-2,
    "query_ratio": 0.05,
    "sample_size": 256,
    "batch_size": 32,
    "quantization_factor": 100,
    "max_seq_length": None,
}


class RegularizerControlError(ValueError):
    """Raised when the regularizer control config or its runtime preconditions are invalid."""


@dataclass(frozen=True)
class DensityControlConfig:
    reference_model: str | None
    reference_dims: float | None
    band: tuple[float, float]
    interval_steps: int
    factor: float
    min_document_weight: float
    max_document_weight: float
    query_ratio: float | None
    sample_size: int
    batch_size: int
    quantization_factor: int
    max_seq_length: int | None
    start_step: int | None = None


@dataclass(frozen=True)
class RegularizerControlConfig:
    warmup_steps: int
    density: DensityControlConfig | None


# --------------------------------------------------------------------------------------------------
# Pure rules


def quadratic_warmup_weight(step: int, warmup_steps: int, target: float) -> float:
    """``target * min(1, (step / warmup_steps) ** 2)``; no warm-up when ``warmup_steps <= 0``."""
    if warmup_steps <= 0:
        return float(target)
    step = max(int(step), 0)
    return float(target) * min(1.0, (step / warmup_steps) ** 2)


def next_document_weight(
    current: float,
    density: float,
    reference: float,
    band: tuple[float, float],
    factor: float,
    bounds: tuple[float, float],
) -> float:
    """One controller step: raise the weight above the band, lower it below, keep it inside."""
    if reference <= 0:
        raise RegularizerControlError(f"Reference density must be positive, got {reference}.")
    ratio = density / reference
    if ratio > band[1]:
        return min(current * factor, bounds[1])
    if ratio < band[0]:
        return max(current / factor, bounds[0])
    return current


# --------------------------------------------------------------------------------------------------
# Config parsing


def _error(key: str, message: str) -> RegularizerControlError:
    return RegularizerControlError(f"'{CONFIG_PATH}.{key}' {message}")


def _as_int(value: Any, key: str) -> int:
    if isinstance(value, bool):
        raise _error(key, f"must be an integer, got {value!r}.")
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            pass
    raise _error(key, f"must be an integer, got {value!r}.")


def _as_float(value: Any, key: str) -> float:
    if isinstance(value, bool):
        raise _error(key, f"must be a number, got {value!r}.")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise _error(key, f"must be a number, got {value!r}.") from exc
    if not math.isfinite(result):
        raise _error(key, f"must be a finite number, got {value!r}.")
    return result


def _positive_int(raw: Mapping[str, Any], key: str, prefix: str) -> int:
    value = _as_int(raw[key], f"{prefix}{key}")
    if value <= 0:
        raise _error(f"{prefix}{key}", f"must be positive, got {value}.")
    return value


def _reject_unknown_keys(raw: Mapping[str, Any], allowed: frozenset[str], where: str) -> None:
    unknown = sorted(str(key) for key in raw if key not in allowed)
    if unknown:
        raise RegularizerControlError(
            f"Unknown key(s) in '{where}': {', '.join(unknown)}. Allowed: {', '.join(sorted(allowed))}."
        )


def _parse_density(raw: Any) -> DensityControlConfig | None:
    if raw is None or raw is False:
        return None
    if not isinstance(raw, Mapping):
        raise _error("density", f"must be a mapping or null, got {type(raw).__name__}.")
    _reject_unknown_keys(raw, _DENSITY_KEYS, f"{CONFIG_PATH}.density")
    values = {**_DENSITY_DEFAULTS, **raw}
    prefix = "density."

    reference_model = values.get("reference_model")
    reference_dims = values.get("reference_dims")
    if (reference_model is None) == (reference_dims is None):
        raise _error("density", "needs exactly one of 'reference_model' and 'reference_dims'.")
    if reference_model is not None:
        if not isinstance(reference_model, str) or not reference_model.strip():
            raise _error(f"{prefix}reference_model", "must be a non-empty path or Hugging Face model id.")
        reference_model = reference_model.strip()
    if reference_dims is not None:
        reference_dims = _as_float(reference_dims, f"{prefix}reference_dims")
        if reference_dims <= 0:
            raise _error(f"{prefix}reference_dims", f"must be positive, got {reference_dims}.")

    band_raw = values["band"]
    if isinstance(band_raw, (str, bytes)) or not isinstance(band_raw, (list, tuple)) or len(band_raw) != 2:
        raise _error(f"{prefix}band", f"must be two numbers [low, high], got {band_raw!r}.")
    low = _as_float(band_raw[0], f"{prefix}band")
    high = _as_float(band_raw[1], f"{prefix}band")
    if not 0 < low < high:
        raise _error(f"{prefix}band", f"must satisfy 0 < low < high, got [{low}, {high}].")

    factor = _as_float(values["factor"], f"{prefix}factor")
    if factor <= 1:
        raise _error(f"{prefix}factor", f"must be greater than 1, got {factor}.")

    min_weight = _as_float(values["min_document_weight"], f"{prefix}min_document_weight")
    max_weight = _as_float(values["max_document_weight"], f"{prefix}max_document_weight")
    if not 0 < min_weight <= max_weight:
        raise _error(
            prefix.rstrip("."),
            f"needs 0 < min_document_weight <= max_document_weight, got {min_weight} and {max_weight}.",
        )

    query_ratio = values["query_ratio"]
    if query_ratio is not None:
        query_ratio = _as_float(query_ratio, f"{prefix}query_ratio")
        if query_ratio <= 0:
            raise _error(f"{prefix}query_ratio", f"must be positive or null, got {query_ratio}.")

    max_seq_length = values["max_seq_length"]
    if max_seq_length is not None:
        max_seq_length = _positive_int(values, "max_seq_length", prefix)
    start_step = values.get("start_step")
    if start_step is not None:
        start_step = _as_int(start_step, f"{prefix}start_step")
        if start_step < 0:
            raise _error(f"{prefix}start_step", "must be >= 0.")

    return DensityControlConfig(
        reference_model=reference_model,
        reference_dims=reference_dims,
        band=(low, high),
        interval_steps=_positive_int(values, "interval_steps", prefix),
        factor=factor,
        min_document_weight=min_weight,
        max_document_weight=max_weight,
        query_ratio=query_ratio,
        sample_size=_positive_int(values, "sample_size", prefix),
        batch_size=_positive_int(values, "batch_size", prefix),
        quantization_factor=_positive_int(values, "quantization_factor", prefix),
        max_seq_length=max_seq_length,
        start_step=start_step,
    )


def parse_regularizer_control(backend_config: Mapping[str, Any]) -> RegularizerControlConfig | None:
    """Read ``regularizer_control`` from the merged backend config; None keeps constant weights."""
    raw = backend_config.get(CONFIG_KEY)
    if raw is None or raw is False:
        return None
    if raw is True:
        raw = {}
    if not isinstance(raw, Mapping):
        raise RegularizerControlError(f"'{CONFIG_PATH}' must be a mapping or null, got {type(raw).__name__}.")
    _reject_unknown_keys(raw, _TOP_LEVEL_KEYS, CONFIG_PATH)
    warmup_steps = _as_int(raw.get("warmup_steps", 0), "warmup_steps")
    if warmup_steps < 0:
        raise _error("warmup_steps", f"must be >= 0, got {warmup_steps}.")
    return RegularizerControlConfig(warmup_steps=warmup_steps, density=_parse_density(raw.get("density")))


def describe_regularizer_control(config: RegularizerControlConfig) -> str:
    density = config.density
    if density is None:
        density_text = "off"
    else:
        reference = density.reference_model if density.reference_model is not None else f"{density.reference_dims:g} dims"
        density_text = (
            f"reference={reference}, band={list(density.band)}, every {density.interval_steps} steps, "
            f"factor={density.factor:g}, bounds=[{density.min_document_weight:g}, {density.max_document_weight:g}], "
            f"query_ratio={density.query_ratio}, sample={density.sample_size}"
        )
    return f"warmup_steps={config.warmup_steps}, density={density_text}"


# --------------------------------------------------------------------------------------------------
# Density measurement (shared with SpladeActivationStatsCallback)


def unique_texts(values: list[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def even_sample(values: list[str], sample_size: int) -> list[str]:
    values = unique_texts(values)
    if len(values) <= sample_size:
        return values
    if sample_size == 1:
        return [values[0]]
    step = (len(values) - 1) / (sample_size - 1)
    return [values[round(idx * step)] for idx in range(sample_size)]


def batch_texts(values: list[str], batch_size: int) -> list[list[str]]:
    return [values[idx : idx + batch_size] for idx in range(0, len(values), batch_size)]


def as_dense_tensor(embeddings: Any, torch: Any) -> Any:
    if isinstance(embeddings, list):
        tensors = [as_dense_tensor(value, torch) for value in embeddings]
        return torch.stack(tensors)
    if not isinstance(embeddings, torch.Tensor):
        return torch.as_tensor(embeddings)
    if embeddings.is_sparse:
        return embeddings.to_dense()
    return embeddings


def count_active_dims(
    model: Any,
    texts: list[str],
    *,
    batch_size: int,
    encoder_name: str,
    quantization_factor: int,
) -> list[int]:
    """Active dimensions per text: ``round(w * quantization_factor) > 0`` (the PIRB sparsity measure).

    The model is put in eval mode for the measurement and returned to train mode afterwards when it
    was training.
    """
    import torch

    counts: list[int] = []
    was_training = bool(getattr(model, "training", False))
    if hasattr(model, "eval"):
        model.eval()
    try:
        for batch in batch_texts(texts, batch_size):
            encoder = getattr(model, encoder_name, None) or getattr(model, "encode")
            embeddings = encoder(
                batch,
                batch_size=len(batch),
                show_progress_bar=False,
                convert_to_tensor=True,
                convert_to_sparse_tensor=False,
                save_to_cpu=False,
            )
            tensor = as_dense_tensor(embeddings, torch)
            # Keep the dense vocabulary-sized embeddings on the accelerator. Moving only
            # the final per-example counts avoids transferring and quantizing roughly
            # sample_size * vocab_size floats on the CPU for every stats interval.
            quantized = torch.round(tensor.float() * quantization_factor)
            counts.extend((quantized > 0).sum(dim=-1).cpu().tolist())
            del embeddings, tensor, quantized
    finally:
        if was_training and hasattr(model, "train"):
            model.train()
    return [int(value) for value in counts]


def _mean(values: list[int]) -> float:
    if not values:
        raise RegularizerControlError("Density measurement returned no values (empty document sample).")
    return sum(values) / len(values)


def _model_device(model: Any) -> str | None:
    device = getattr(model, "device", None)
    return None if device is None else str(device)


def _release_accelerator_memory() -> None:
    gc.collect()
    try:
        import torch
    except ModuleNotFoundError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# --------------------------------------------------------------------------------------------------
# Callback


class RegularizerControlCallback:
    """Sets ``loss.document_regularizer_weight`` / ``loss.query_regularizer_weight`` before every step.

    Duck-typed Transformers callback: every ``on_*`` event it does not define is a no-op.
    """

    def __init__(
        self,
        loss: Any,
        model: Any,
        document_texts: list[str],
        config: RegularizerControlConfig,
        target_document_weight: float,
        target_query_weight: float | None,
        prefix: str = DEFAULT_PREFIX,
        *,
        density_fn: Callable[[], float] | None = None,
        reference_fn: Callable[[], float] | None = None,
        main_process: bool | None = None,
    ) -> None:
        if not hasattr(loss, "document_regularizer_weight"):
            raise RegularizerControlError(
                f"regularizer_control needs a SPLADE loss with a 'document_regularizer_weight' attribute; "
                f"got {type(loss).__name__}."
            )
        if target_query_weight is not None and not hasattr(loss, "query_regularizer_weight"):
            raise RegularizerControlError(
                f"regularizer_control needs a 'query_regularizer_weight' attribute on {type(loss).__name__}."
            )
        if config.density is not None and not document_texts and density_fn is None:
            raise RegularizerControlError("regularizer_control.density needs a non-empty sample of training documents.")
        self.loss = loss
        self.model = model
        self.document_texts = list(document_texts)
        self.config = config
        self.target_document_weight = float(target_document_weight)
        self.target_query_weight = None if target_query_weight is None else float(target_query_weight)
        self.prefix = prefix.rstrip("/")
        self.density_fn = density_fn if density_fn is not None else self._measure_model_density
        self.reference_fn = reference_fn if reference_fn is not None else self._measure_reference_density
        self.main_process = is_main_process() if main_process is None else bool(main_process)

        self.document_weight = self.target_document_weight
        self.query_weight = self.target_query_weight
        self.reference_dims: float | None = config.density.reference_dims if config.density is not None else None
        self.last_density: float | None = None
        self.last_control_step: int | None = None
        self.history: list[dict[str, Any]] = []
        self._history_path: Path | None = None

    def __getattr__(self, name: str) -> Any:
        if name.startswith("on_"):
            def _noop(args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
                return control

            return _noop
        raise AttributeError(name)

    # -- weights ---------------------------------------------------------------------------------

    def weights_at(self, step: int) -> tuple[float, float | None]:
        """Weights the loss uses at optimizer step ``step`` (``state.global_step`` before the step)."""
        warmup_steps = self.config.warmup_steps
        if step < warmup_steps:
            document = quadratic_warmup_weight(step, warmup_steps, self.document_weight)
            query = None if self.query_weight is None else quadratic_warmup_weight(step, warmup_steps, self.query_weight)
            return document, query
        return self.document_weight, self.query_weight

    def _apply(self, step: int) -> None:
        document, query = self.weights_at(step)
        self.loss.document_regularizer_weight = document
        if query is not None:
            self.loss.query_regularizer_weight = query

    # -- events ----------------------------------------------------------------------------------

    def on_train_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        step = _global_step(state)
        self._history_path = _history_path(args)
        density = self.config.density
        if density is not None and self.reference_dims is None:
            reference = float(self.reference_fn())
            if not reference > 0:
                raise RegularizerControlError(
                    f"The reference model {density.reference_model!r} has no active dimensions on the sample."
                )
            self.reference_dims = reference
        resumed_from = self._restore(step) if step > 0 else None
        self._apply(step)
        document, query = self.weights_at(step)
        self._record(step, "train_begin", document_weight=document, query_weight=query, resumed_from=resumed_from)
        if self.main_process:
            message = f"[INFO] Regularizer control at step {step}: document_weight={document:g}"
            if query is not None:
                message += f", query_weight={query:g}"
            if self.reference_dims is not None:
                message += f", document_dims_reference={self.reference_dims:.1f}"
            if resumed_from is not None:
                message += f", controller state restored from step {resumed_from}"
            print(message + ".", flush=True)
        return control

    def on_step_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        self._apply(_global_step(state))
        return control

    def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        step = _global_step(state)
        if self.config.warmup_steps > 0 and step == self.config.warmup_steps:
            document, query = self.weights_at(step)
            self._record(step, "warmup_end", document_weight=document, query_weight=query)
            if self.main_process:
                print(
                    f"[INFO] Regularizer control: warm-up finished at step {step}, document_weight={document:g}"
                    + (f", query_weight={query:g}" if query is not None else "")
                    + ".",
                    flush=True,
                )
        if not self._should_control(step):
            return control
        density_config = self.config.density
        assert density_config is not None and self.reference_dims is not None
        density = float(self.density_fn())
        previous = self.document_weight
        self.document_weight = next_document_weight(
            previous,
            density,
            self.reference_dims,
            density_config.band,
            density_config.factor,
            (density_config.min_document_weight, density_config.max_document_weight),
        )
        if density_config.query_ratio is not None and self.query_weight is not None:
            self.query_weight = density_config.query_ratio * self.document_weight
        self.last_density = density
        self.last_control_step = step
        self._apply(step)
        ratio = density / self.reference_dims
        action = "up" if self.document_weight > previous else "down" if self.document_weight < previous else "hold"
        self._record(
            step,
            "control",
            document_weight=self.document_weight,
            query_weight=self.query_weight,
            document_dims_mean=density,
            document_dims_ratio=ratio,
            action=action,
        )
        if self.main_process:
            print(
                f"[INFO] Regularizer control at step {step}: document_dims_mean={density:.1f} "
                f"(ratio {ratio:.3f} to {self.reference_dims:.1f}), document_weight {previous:g} -> "
                f"{self.document_weight:g} ({action})"
                + (f", query_weight={self.query_weight:g}" if self.query_weight is not None else "")
                + ".",
                flush=True,
            )
        return control

    def on_log(self, args: Any, state: Any, control: Any, logs: dict[str, Any] | None = None, **kwargs: Any) -> Any:
        metrics = self.metrics(_global_step(state))
        if logs is not None:
            logs.update(metrics)
        merge_into_log_history(state, metrics)
        return control

    def on_train_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        step = _global_step(state)
        document, query = self.weights_at(step)
        self._record(step, "train_end", document_weight=document, query_weight=query)
        return control

    # -- helpers ---------------------------------------------------------------------------------

    def _should_control(self, step: int) -> bool:
        density = self.config.density
        return (
            density is not None
            and step >= (self.config.warmup_steps if density.start_step is None else density.start_step)
            and step > 0
            and step % density.interval_steps == 0
            and step != self.last_control_step
        )

    def metrics(self, step: int) -> dict[str, float]:
        document, query = self.weights_at(step)
        metrics = {f"{self.prefix}/document_weight": document}
        if query is not None:
            metrics[f"{self.prefix}/query_weight"] = query
        if self.last_density is not None:
            metrics[f"{self.prefix}/document_dims_mean"] = self.last_density
        if self.reference_dims is not None:
            metrics[f"{self.prefix}/document_dims_reference"] = self.reference_dims
            if self.last_density is not None:
                metrics[f"{self.prefix}/document_dims_ratio"] = self.last_density / self.reference_dims
        return metrics

    def _record(self, step: int, event: str, **values: Any) -> None:
        entry = {"step": int(step), "event": event, **{key: value for key, value in values.items() if value is not None}}
        self.history.append(entry)
        self._write_history()

    def _write_history(self) -> None:
        if not self.main_process or self._history_path is None:
            return
        payload = {
            "config": asdict(self.config),
            "target_document_weight": self.target_document_weight,
            "target_query_weight": self.target_query_weight,
            "document_dims_reference": self.reference_dims,
            "events": self.history,
        }
        path = self._history_path
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = path.with_name(path.name + ".tmp")
            tmp_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            os.replace(tmp_path, path)
        except OSError as exc:
            print(f"[WARN] Could not write {path}: {exc}", flush=True)

    def _restore(self, step: int) -> int | None:
        """On resume, take the controller state of the last control at or before ``step``."""
        if self.config.density is None or self._history_path is None or not self._history_path.is_file():
            return None
        try:
            payload = json.loads(self._history_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        events = [event for event in payload.get("events", []) if isinstance(event, dict)]
        kept = [event for event in events if int(event.get("step", -1)) <= step]
        controls = [event for event in kept if event.get("event") == "control"]
        self.history = kept
        if not controls:
            return None
        last = controls[-1]
        self.document_weight = float(last["document_weight"])
        if self.query_weight is not None and last.get("query_weight") is not None:
            self.query_weight = float(last["query_weight"])
        if last.get("document_dims_mean") is not None:
            self.last_density = float(last["document_dims_mean"])
        self.last_control_step = int(last["step"])
        return self.last_control_step

    def _measure_model_density(self) -> float:
        density = self.config.density
        assert density is not None
        model = self.model
        original_length = getattr(model, "max_seq_length", None)
        override = density.max_seq_length is not None and original_length is not None
        if override:
            model.max_seq_length = density.max_seq_length
        try:
            counts = count_active_dims(
                model,
                self.document_texts,
                batch_size=density.batch_size,
                encoder_name="encode_document",
                quantization_factor=density.quantization_factor,
            )
        finally:
            if override:
                model.max_seq_length = original_length
        return _mean(counts)

    def _measure_reference_density(self) -> float:
        density = self.config.density
        assert density is not None and density.reference_model is not None
        from sentence_transformers.sparse_encoder import SparseEncoder

        reference = SparseEncoder(density.reference_model, device=_model_device(self.model))
        max_seq_length = density.max_seq_length or getattr(self.model, "max_seq_length", None)
        if max_seq_length is not None:
            reference.max_seq_length = int(max_seq_length)
        try:
            counts = count_active_dims(
                reference,
                self.document_texts,
                batch_size=density.batch_size,
                encoder_name="encode_document",
                quantization_factor=density.quantization_factor,
            )
        finally:
            del reference
            _release_accelerator_memory()
        return _mean(counts)


def _global_step(state: Any) -> int:
    return int(getattr(state, "global_step", 0) or 0)


def merge_into_log_history(state: Any, metrics: Mapping[str, Any]) -> None:
    """Add callback metrics to the ``log_history`` entry of the current step.

    ``Trainer.log`` appends a copy of ``logs`` to ``state.log_history`` before it calls ``on_log``, so
    keys a callback adds to ``logs`` never reach ``trainer_state.json`` on their own.
    """
    if not metrics:
        return
    log_history = getattr(state, "log_history", None)
    if isinstance(log_history, list) and log_history and isinstance(log_history[-1], dict):
        entry = log_history[-1]
        if entry.get("step") == _global_step(state):
            entry.update(metrics)


def _history_path(args: Any) -> Path | None:
    output_dir = getattr(args, "output_dir", None)
    if not output_dir:
        return None
    return Path(str(output_dir)) / HISTORY_FILE_NAME


def document_sample(rows: list[dict[str, Any]], sample_size: int) -> list[str]:
    """Positives plus negatives, evenly sampled (same rule as SpladeActivationStatsCallback)."""
    candidates = [row["positive"] for row in rows if row.get("positive")]
    for row in rows:
        for key, value in row.items():
            if key.startswith("negative_") and value:
                candidates.append(value)
    return even_sample(candidates, sample_size)


def build_regularizer_control_callback(
    loss: Any,
    model: Any,
    rows: list[dict[str, Any]],
    backend_config: Mapping[str, Any],
) -> RegularizerControlCallback | None:
    """The callback for ``regularizer_control``, or None when the key is absent or null.

    Built on every rank: each rank owns a copy of the loss, so every rank has to follow the same
    weights. Only the main process prints and writes ``regularizer_control.json``.
    """
    config = parse_regularizer_control(backend_config)
    if config is None:
        return None
    document_texts = document_sample(rows, config.density.sample_size) if config.density is not None else []
    raw_query_weight = backend_config.get("query_regularizer_weight", 5e-5)
    return RegularizerControlCallback(
        loss,
        model,
        document_texts,
        config,
        target_document_weight=float(backend_config.get("document_regularizer_weight", 3e-5)),
        target_query_weight=None if raw_query_weight is None else float(raw_query_weight),
    )


def install_regularizer_control_callback(trainer: Any, callback: RegularizerControlCallback) -> str:
    """Put the controller where sentence-transformers keeps its own weight scheduler.

    ``SparseEncoderTrainer`` inserts ``SpladeRegularizerWeightSchedulerCallback`` at position 1 of the
    callback list (before the reporting callbacks). The controller replaces it there, so it alone
    sets the weights, and the keys it adds in ``on_log`` reach W&B and the console, which only see
    ``logs`` mutated by callbacks that run before them. Returns a short description of what happened.
    """
    handler = getattr(trainer, "callback_handler", None)
    callbacks = getattr(handler, "callbacks", None)
    if isinstance(callbacks, list):
        for index, existing in enumerate(callbacks):
            if type(existing).__name__ in BUILTIN_WEIGHT_SCHEDULER_NAMES:
                callbacks[index] = callback
                return f"replaced {type(existing).__name__}"
        callbacks.insert(min(1, len(callbacks)), callback)
        return "inserted"
    trainer.add_callback(callback)
    return "added"
