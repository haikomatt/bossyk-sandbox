"""Training backend for the two detector families (phase-detector-training.md
build-order step 4): (a) DeBERTa-v3-small full fine-tune, (b)
Qwen2.5-0.5B-Instruct LoRA (PEFT). Binary classification (violation vs not)
on decision text, with class-weighting (not resampling) for imbalance, and
Amendment 3's calibration-slice protocol wired through a single orchestration
function shared by both families.

`torch`/`transformers`/`peft` are imported LAZILY, inside `HFTrainBackend`'s
methods, exactly like `guardrail.model_backed.load_injection_classifier` --
so this module and the default test suite never require the pod-only
`detector-train` extra. `TrainBackend` is a `Protocol` so
`run_training_cell` -- the part that actually encodes the pre-registered
protocol (calib slice, class weights, temperature fit, manifest fields) --
is hermetically testable against `StubTrainBackend`, which never touches a
GPU. That is the brief's "hermetic build (no GPU, no spend)" for training:
the orchestration logic is real and tested; only the model fit/score calls
are swapped for a stub until the pod exists.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from bossyk_sandbox.detector.calib_slice import calibration_slice_split
from bossyk_sandbox.detector.temperature import fit_temperature
from bossyk_sandbox.interp.corpus_assembly import Record

FAMILIES: tuple[str, ...] = ("deberta", "qwen_lora")
SEEDS: tuple[int, ...] = (0, 1, 2)

MODEL_NAME_BY_FAMILY: dict[str, str] = {
    "deberta": "microsoft/deberta-v3-small",
    "qwen_lora": "Qwen/Qwen2.5-0.5B-Instruct",
}


def decision_text(row: Record) -> str:
    """Same text convention as the frozen baselines/judge
    (`scripts/detector_baselines.py::decision_text`) -- context prompt then
    the action taken, since the policy label is about the action, not just
    the request."""
    return f"{row['prompt']}\n\nACTION:\n{row['action']}"


def class_weights(y: list[bool]) -> dict[bool, float]:
    """Inverse-frequency class weights for imbalance handling (brief: "Class
    imbalance via class weighting (do not resample the data)"). The
    standard balanced-weight formula (`n / (n_classes * count[c])`,
    equivalent to sklearn's `class_weight="balanced"`), reimplemented here
    to avoid a new dependency. A single-class split has nothing to weight
    against, so both weights fall back to 1.0."""
    n = len(y)
    if n == 0:
        return {True: 1.0, False: 1.0}
    n_pos = sum(1 for v in y if v)
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return {True: 1.0, False: 1.0}
    return {True: n / (2.0 * n_pos), False: n / (2.0 * n_neg)}


@dataclass(frozen=True)
class TrainConfig:
    family: str
    train_domain: str
    seed: int
    slice_seed: int
    train_rows: list[Record]  # the domain's full v2 TRAIN split; 85/15 slice happens inside
    val_rows: list[Record]  # the domain's v2 VAL split (early stopping, loss only)
    output_dir: Path
    max_epochs: int = 10
    early_stopping_patience: int = 2
    per_device_batch_size: int = 16
    learning_rate: float = 2e-5
    lora_r: int = 16
    lora_alpha: int = 32


@dataclass(frozen=True)
class TrainResult:
    family: str
    train_domain: str
    seed: int
    slice_seed: int
    checkpoint_dir: Path
    final_val_loss: float
    early_stop_epoch: int
    class_weights: dict[str, float]
    calib_temperature: float
    wall_seconds: float
    n_train: int
    n_calib: int
    n_val: int
    train_violation_base_rate: float

    def to_manifest_dict(self) -> dict[str, Any]:
        """JSON-serialisable training-manifest row (brief step 7: family,
        train domain, seed, slice seed, epochs, early-stopping epoch, class
        weights, temperature fitted, wall time)."""
        return {
            "family": self.family,
            "train_domain": self.train_domain,
            "seed": self.seed,
            "slice_seed": self.slice_seed,
            "checkpoint_dir": str(self.checkpoint_dir),
            "final_val_loss": self.final_val_loss,
            "early_stop_epoch": self.early_stop_epoch,
            "class_weights": self.class_weights,
            "calib_temperature": self.calib_temperature,
            "wall_seconds": self.wall_seconds,
            "n_train": self.n_train,
            "n_calib": self.n_calib,
            "n_val": self.n_val,
            "train_violation_base_rate": self.train_violation_base_rate,
        }


@dataclass(frozen=True)
class BackendTrainOutput:
    checkpoint_dir: Path
    final_val_loss: float
    early_stop_epoch: int


class TrainBackend(Protocol):
    def train(
        self,
        *,
        family: str,
        train_rows: list[Record],
        val_rows: list[Record],
        weights: dict[bool, float],
        output_dir: Path,
        seed: int,
        max_epochs: int,
        early_stopping_patience: int,
        per_device_batch_size: int,
        learning_rate: float,
        lora_r: int,
        lora_alpha: int,
    ) -> BackendTrainOutput: ...

    def score(self, *, family: str, checkpoint_dir: Path, rows: list[Record]) -> list[float]:
        """Raw (pre-sigmoid, pre-temperature) logits for `rows`, aligned by
        index -- the caller applies `temperature.apply_temperature`."""
        ...


def run_training_cell(cfg: TrainConfig, backend: TrainBackend) -> TrainResult:
    """The pre-registered protocol, backend-agnostic: Amendment 3's
    calib-slice split of `cfg.train_rows` (seeded on `cfg.slice_seed`),
    class weights fit on the 85% train-proper bucket, delegate the actual
    fit to `backend.train`, then fit temperature on the backend's scores
    over the calibration slice. This function is the thing the brief calls
    "Amendment-3 slice logic as a tested, seeded function" -- tested here
    via `StubTrainBackend`, reused unchanged by the real pod run."""
    start = time.monotonic()
    slice_result = calibration_slice_split(cfg.train_rows, seed=cfg.slice_seed)
    y_train_proper = [bool(r["is_violation"]) for r in slice_result.train]
    weights = class_weights(y_train_proper)

    backend_output = backend.train(
        family=cfg.family,
        train_rows=slice_result.train,
        val_rows=cfg.val_rows,
        weights=weights,
        output_dir=cfg.output_dir,
        seed=cfg.seed,
        max_epochs=cfg.max_epochs,
        early_stopping_patience=cfg.early_stopping_patience,
        per_device_batch_size=cfg.per_device_batch_size,
        learning_rate=cfg.learning_rate,
        lora_r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
    )

    if slice_result.calib:
        calib_logits = backend.score(
            family=cfg.family, checkpoint_dir=backend_output.checkpoint_dir, rows=slice_result.calib
        )
        y_calib = np.array([bool(r["is_violation"]) for r in slice_result.calib], dtype=bool)
        temperature = fit_temperature(np.array(calib_logits, dtype=np.float64), y_calib)
    else:
        # No calib rows at all (pathological tiny domain) -- no-op scaling,
        # not a fabricated fit.
        temperature = 1.0

    wall = time.monotonic() - start
    n_train = len(slice_result.train)
    n_pos = sum(y_train_proper)
    return TrainResult(
        family=cfg.family,
        train_domain=cfg.train_domain,
        seed=cfg.seed,
        slice_seed=cfg.slice_seed,
        checkpoint_dir=backend_output.checkpoint_dir,
        final_val_loss=backend_output.final_val_loss,
        early_stop_epoch=backend_output.early_stop_epoch,
        class_weights={"violation": weights[True], "compliant": weights[False]},
        calib_temperature=temperature,
        wall_seconds=wall,
        n_train=n_train,
        n_calib=len(slice_result.calib),
        n_val=len(cfg.val_rows),
        train_violation_base_rate=(n_pos / n_train if n_train else float("nan")),
    )


@dataclass
class StubTrainBackend:
    """A CPU/no-deps `TrainBackend` for hermetic smoke tests and the CLI's
    `--dry-run` mode (brief Phase A item 4: "Smoke the full train+score
    pipeline locally with a tiny model stub or 1-batch dry-run mode
    (hermetic, CPU) before any pod exists."). Fits nothing; "scores" rows
    with a deterministic seeded function of the row id and label so tests
    can assert real shape/plumbing (calib split, class weights, temperature
    fit, manifest fields, output files) without torch installed."""

    def train(
        self,
        *,
        family: str,
        train_rows: list[Record],
        val_rows: list[Record],
        weights: dict[bool, float],
        output_dir: Path,
        seed: int,
        **_: Any,
    ) -> BackendTrainOutput:
        ckpt = Path(output_dir) / f"{family}-seed{seed}" / "final"
        ckpt.mkdir(parents=True, exist_ok=True)
        (ckpt / "stub_manifest.json").write_text(
            json.dumps(
                {
                    "family": family,
                    "n_train": len(train_rows),
                    "n_val": len(val_rows),
                    "weights": {str(k): v for k, v in weights.items()},
                    "seed": seed,
                }
            )
        )
        return BackendTrainOutput(checkpoint_dir=ckpt, final_val_loss=0.5, early_stop_epoch=1)

    def score(self, *, family: str, checkpoint_dir: Path, rows: list[Record]) -> list[float]:
        out = []
        for r in rows:
            base = 3.0 if r.get("is_violation") else -3.0
            noise = (hash((str(r.get("row_id", "")), family)) % 1000) / 1000.0 - 0.5
            out.append(base + noise)
        return out


def _find_resumable_checkpoint(output_dir: Path) -> str | None:
    """Latest `checkpoint-<step>` subdir under `output_dir`, if any -- HF
    `Trainer`'s own on-disk checkpoint format, so `trainer.train(
    resume_from_checkpoint=...)` picks up exactly where a prior (possibly
    budget-terminated) run left off. Returns `None` (fresh start) if
    `output_dir` has no checkpoints yet."""
    if not output_dir.exists():
        return None
    candidates = []
    for child in output_dir.iterdir():
        if child.is_dir() and child.name.startswith("checkpoint-"):
            try:
                step = int(child.name.split("-")[-1])
            except ValueError:
                continue
            candidates.append((step, child))
    if not candidates:
        return None
    candidates.sort(key=lambda pair: pair[0])
    return str(candidates[-1][1])


def _epoch_from_checkpoint(checkpoint_dir: str, *, default: int) -> int:
    state_path = Path(checkpoint_dir) / "trainer_state.json"
    if not state_path.exists():
        return default
    try:
        state = json.loads(state_path.read_text())
        return int(round(float(state.get("epoch", default))))
    except (ValueError, json.JSONDecodeError):
        return default


@dataclass
class HFTrainBackend:
    """Real backend: HF `Trainer` full fine-tune (DeBERTa-v3-small) or
    PEFT/LoRA (Qwen2.5-0.5B-Instruct), binary classification over decision
    text, class-weighted cross-entropy loss. POD ONLY -- `train`/`score`
    lazily import `torch`/`transformers`/`peft` so this class can be
    imported (e.g. for `isinstance` checks in the CLI) without those deps
    present."""

    max_length: int = 512
    eval_batch_size: int = 32

    def train(
        self,
        *,
        family: str,
        train_rows: list[Record],
        val_rows: list[Record],
        weights: dict[bool, float],
        output_dir: Path,
        seed: int,
        max_epochs: int,
        early_stopping_patience: int,
        per_device_batch_size: int,
        learning_rate: float,
        lora_r: int,
        lora_alpha: int,
    ) -> BackendTrainOutput:
        import torch
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            EarlyStoppingCallback,
            Trainer,
            TrainingArguments,
            set_seed,
        )

        if family not in MODEL_NAME_BY_FAMILY:
            raise ValueError(f"unknown family: {family!r}")
        set_seed(seed)
        model_name = MODEL_NAME_BY_FAMILY[family]

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
        if getattr(model.config, "pad_token_id", None) is None:
            model.config.pad_token_id = tokenizer.pad_token_id

        if family == "qwen_lora":
            from peft import LoraConfig, TaskType, get_peft_model

            lora_config = LoraConfig(
                task_type=TaskType.SEQ_CLS,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=0.05,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            )
            model = get_peft_model(model, lora_config)

        class _Dataset(torch.utils.data.Dataset):  # type: ignore[misc]
            def __init__(self, encodings: Any, labels: list[int]) -> None:
                self.encodings = encodings
                self.labels = torch.tensor(labels, dtype=torch.long)

            def __len__(self) -> int:
                return len(self.labels)

            def __getitem__(self, idx: int) -> dict[str, Any]:
                item = {k: v[idx] for k, v in self.encodings.items()}
                item["labels"] = self.labels[idx]
                return item

        train_texts = [decision_text(r) for r in train_rows]
        train_labels = [int(bool(r["is_violation"])) for r in train_rows]
        train_enc = tokenizer(
            train_texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        train_ds = _Dataset(train_enc, train_labels)

        eval_ds = None
        if val_rows:
            val_texts = [decision_text(r) for r in val_rows]
            val_labels = [int(bool(r["is_violation"])) for r in val_rows]
            val_enc = tokenizer(
                val_texts,
                truncation=True,
                padding=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            eval_ds = _Dataset(val_enc, val_labels)

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        args_kwargs: dict[str, Any] = dict(
            output_dir=str(output_dir),
            num_train_epochs=max_epochs,
            per_device_train_batch_size=per_device_batch_size,
            per_device_eval_batch_size=per_device_batch_size,
            learning_rate=learning_rate,
            seed=seed,
            save_total_limit=2,
            logging_steps=50,
            report_to=[],
        )
        if eval_ds is not None:
            args_kwargs.update(
                eval_strategy="epoch",
                save_strategy="epoch",
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",
                greater_is_better=False,
            )
        else:
            args_kwargs.update(save_strategy="epoch")
        training_args = TrainingArguments(**args_kwargs)

        weight_tensor = torch.tensor([weights[False], weights[True]], dtype=torch.float32)

        class _WeightedTrainer(Trainer):  # type: ignore[misc]
            def compute_loss(
                self,
                model: Any,
                inputs: dict[str, Any],
                return_outputs: bool = False,
                **kwargs: Any,
            ) -> Any:
                labels = inputs.pop("labels")
                outputs = model(**inputs)
                logits = outputs.logits
                loss_fct = torch.nn.CrossEntropyLoss(weight=weight_tensor.to(logits.device))
                loss = loss_fct(logits.view(-1, 2), labels.view(-1))
                return (loss, outputs) if return_outputs else loss

        callbacks = (
            [EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)]
            if eval_ds is not None
            else []
        )

        trainer = _WeightedTrainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            callbacks=callbacks,
        )

        resume = _find_resumable_checkpoint(output_dir)
        train_output = trainer.train(resume_from_checkpoint=resume)

        final_val_loss = float("nan")
        early_stop_epoch = int(round(float(train_output.metrics.get("epoch", max_epochs))))
        if eval_ds is not None:
            eval_metrics = trainer.evaluate()
            final_val_loss = float(eval_metrics.get("eval_loss", float("nan")))
            if trainer.state.best_model_checkpoint:
                early_stop_epoch = _epoch_from_checkpoint(
                    trainer.state.best_model_checkpoint, default=early_stop_epoch
                )

        final_dir = output_dir / "final"
        if family == "qwen_lora":
            merged = trainer.model.merge_and_unload()
            merged.save_pretrained(str(final_dir))
        else:
            trainer.save_model(str(final_dir))
        tokenizer.save_pretrained(str(final_dir))

        return BackendTrainOutput(
            checkpoint_dir=final_dir,
            final_val_loss=final_val_loss,
            early_stop_epoch=early_stop_epoch,
        )

    def score(self, *, family: str, checkpoint_dir: Path, rows: list[Record]) -> list[float]:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_dir))
        model = AutoModelForSequenceClassification.from_pretrained(str(checkpoint_dir))
        model.eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)

        texts = [decision_text(r) for r in rows]
        logits_out: list[float] = []
        with torch.no_grad():
            for i in range(0, len(texts), self.eval_batch_size):
                batch = texts[i : i + self.eval_batch_size]
                enc = tokenizer(
                    batch,
                    truncation=True,
                    padding=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(device)
                out = model(**enc)
                # Binary logit = logit[violation=1] - logit[compliant=0], so
                # `temperature.apply_temperature`'s sigmoid convention
                # matches a 2-class head without a separate binarisation step.
                diff = (out.logits[:, 1] - out.logits[:, 0]).detach().cpu().tolist()
                logits_out.extend(diff)
        return logits_out
