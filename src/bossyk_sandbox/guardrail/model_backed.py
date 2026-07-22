from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from bossyk_sandbox.guardrail.guardrail import Guardrail, GuardrailStrength, GuardrailVerdict


class InjectionClassifier(Protocol):
    """Matches a HF text-classification model wrapped for injection/
    jailbreak detection -- satisfied by a fake in tests, or the real
    `protectai/deberta-v3-base-prompt-injection-v2` pipeline for the real
    run (see `build_model_backed_guardrail`)."""

    def predict(self, content: str) -> float:
        """Returns P(injection/jailbreak) in [0, 1]."""
        ...


# Lower threshold = flags more. A stricter guardrail lowers the bar to flag,
# so STRICT flags a superset of LEAKY's flags at any fixed classifier output
# -- same nesting property `GradedRuleGuardrail` guarantees via its tiered
# pattern sets (OFF is handled specially in `inspect`, not via this table).
THRESHOLD_BY_STRENGTH: dict[GuardrailStrength, float] = {
    GuardrailStrength.LEAKY: 0.9,
    GuardrailStrength.MODERATE: 0.6,
    GuardrailStrength.STRICT: 0.3,
}


@dataclass
class ModelBackedGuardrail:
    """`Guardrail` backed by a model-based `InjectionClassifier` instead of
    `GradedRuleGuardrail`'s keyword rules -- same interface, swappable in
    `conditions.harness.run_probe_grid`."""

    classifier: InjectionClassifier
    strength: GuardrailStrength = GuardrailStrength.LEAKY

    def inspect(self, content: str) -> GuardrailVerdict:
        if self.strength == GuardrailStrength.OFF:
            return GuardrailVerdict(flagged=False, reason="guardrail off")

        prob = self.classifier.predict(content)
        threshold = THRESHOLD_BY_STRENGTH[self.strength]
        flagged = prob >= threshold
        return GuardrailVerdict(
            flagged=flagged, reason=f"p(injection)={prob:.2f} threshold={threshold}"
        )


INJECTION_CLASSIFIER_MODEL = "protectai/deberta-v3-base-prompt-injection-v2"

# The attack/positive class label the HF classifier emits, plus a generic
# LABEL_1 fallback for models that don't name their classes.
_INJECTION_LABELS = {"INJECTION", "LABEL_1"}


def _positive_injection_prob(pipeline_output: Any) -> float:
    # `pipeline_output` is `Any` because transformers' text-classification
    # pipeline return shape is version-dependent: with `top_k=None` it is
    # either `list[dict]` (one dict per class) or `list[list[dict]]` (each
    # input's class list wrapped once more). Normalise to the flat per-class
    # list, then read the injection class's score.
    scores = pipeline_output
    if scores and isinstance(scores[0], list):
        scores = scores[0]
    for score in scores:
        if str(score["label"]).upper() in _INJECTION_LABELS:
            return float(score["score"])
    raise ValueError(f"no injection label in classifier output: {pipeline_output!r}")


@dataclass
class _PipelineInjectionClassifier:
    """Wraps a transformers text-classification pipeline as an
    `InjectionClassifier`. `pipeline` is `Any` to keep this module free of a
    hard typing dependency on `transformers`."""

    pipeline: Any

    def predict(self, content: str) -> float:
        return _positive_injection_prob(self.pipeline(content))


def load_injection_classifier(model: str = INJECTION_CLASSIFIER_MODEL) -> InjectionClassifier:
    """Real path: load the HF prompt-injection classifier lazily (inside the
    function, not at module import) so this module and the unit-test suite
    never require `transformers`/`torch`, mirroring
    `instruments.drift.build_default_drift_instrument`. The model downloads
    from the HF Hub on first use; not bundled or fetched in tests. Exposed
    separately so a caller (e.g. the H1 benchmark sweep) can load the model
    once and reuse it across several `ModelBackedGuardrail` strengths."""
    from transformers import pipeline

    classifier_pipeline = pipeline("text-classification", model=model, top_k=None)
    return _PipelineInjectionClassifier(pipeline=classifier_pipeline)


def build_model_backed_guardrail(
    strength: GuardrailStrength = GuardrailStrength.LEAKY,
) -> Guardrail:
    return ModelBackedGuardrail(classifier=load_injection_classifier(), strength=strength)
