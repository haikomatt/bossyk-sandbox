from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

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


def build_model_backed_guardrail(
    strength: GuardrailStrength = GuardrailStrength.LEAKY,
) -> Guardrail:
    # Real path: a HF prompt-injection/jailbreak text-classification model,
    # loaded lazily (inside the function, not at module import) so this
    # module -- and the unit test suite -- never require `transformers`/
    # `torch` to be installed, mirroring
    # `instruments.drift.build_default_drift_instrument`'s gated import.
    # Requires the `transformers` package (and its `torch` backend) to be
    # installed in the running environment; the model itself is downloaded
    # from the HF Hub on first use, not bundled or fetched in tests.
    from transformers import pipeline

    classifier_pipeline = pipeline(
        "text-classification",
        model="protectai/deberta-v3-base-prompt-injection-v2",
        top_k=None,
    )

    class _PipelineInjectionClassifier:
        def predict(self, content: str) -> float:
            # `top_k=None` on a single (non-batched) string input returns a
            # flat list of {"label", "score"} dicts, one per class -- not a
            # per-input-wrapped list, since there's only one input here.
            scores = classifier_pipeline(content)
            for score in scores:
                if score["label"].upper() in {"INJECTION", "LABEL_1"}:
                    probability: float = score["score"]
                    return probability
            raise ValueError(f"no INJECTION label in classifier output: {scores!r}")

    return ModelBackedGuardrail(classifier=_PipelineInjectionClassifier(), strength=strength)
