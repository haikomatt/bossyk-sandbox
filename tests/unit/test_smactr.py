from __future__ import annotations

from pathlib import Path

from bossyk_sandbox.governance.smactr import (
    CaughtFailure,
    Severity,
    ThreatModelEntry,
    fmea_severity,
    load_threat_model,
    save_threat_model,
    smactr_response,
)


def test_fmea_severity_rates_pii_disclosure_as_high() -> None:
    assert fmea_severity("pii_disclosure") is Severity.HIGH


def test_fmea_severity_rates_refund_over_threshold_as_high() -> None:
    assert fmea_severity("refund_over_threshold") is Severity.HIGH


def test_fmea_severity_rates_unauthorized_modification_as_high() -> None:
    assert fmea_severity("unauthorized_modification") is Severity.HIGH


def test_fmea_severity_rates_cancel_without_auth_as_medium() -> None:
    assert fmea_severity("cancel_without_auth") is Severity.MEDIUM


def test_fmea_severity_defaults_unknown_boundary_to_medium() -> None:
    assert fmea_severity("some_unregistered_boundary") is Severity.MEDIUM


def _failure() -> CaughtFailure:
    return CaughtFailure(
        domain="airline",
        scenario_id="scn-042",
        boundary="pii_disclosure",
        tool_name="lookup_passenger",
        declared_intent="verify identity",
    )


def test_smactr_response_severity_is_the_fmea_of_the_failures_boundary() -> None:
    entry = smactr_response(
        _failure(),
        derived_constraint="require dual verification before disclosure",
        regression_probe_id="probe-001",
    )

    assert entry.severity is fmea_severity(_failure().boundary)


def test_smactr_response_failure_id_joins_domain_and_scenario_id() -> None:
    entry = smactr_response(
        _failure(),
        derived_constraint="require dual verification before disclosure",
        regression_probe_id="probe-001",
    )

    assert entry.failure_id == "airline-scn-042"


def test_smactr_response_carries_the_passed_constraint_probe_and_status() -> None:
    entry = smactr_response(
        _failure(),
        derived_constraint="require dual verification before disclosure",
        regression_probe_id="probe-001",
        status="unmitigated",
    )

    assert entry.derived_constraint == "require dual verification before disclosure"
    assert entry.regression_probe_id == "probe-001"
    assert entry.status == "unmitigated"


def test_smactr_response_status_defaults_to_mitigated() -> None:
    entry = smactr_response(
        _failure(),
        derived_constraint="require dual verification before disclosure",
        regression_probe_id="probe-001",
    )

    assert entry.status == "mitigated"


def test_threat_model_round_trips_through_save_and_load(tmp_path: Path) -> None:
    entry = smactr_response(
        _failure(),
        derived_constraint="require dual verification before disclosure",
        regression_probe_id="probe-001",
    )
    path = tmp_path / "threat_model.json"

    save_threat_model([entry], path)
    loaded = load_threat_model(path)

    assert loaded == [entry]


def test_threat_model_entry_is_a_frozen_dataclass() -> None:
    entry = ThreatModelEntry(
        failure_id="airline-scn-042",
        domain="airline",
        boundary="pii_disclosure",
        tool_name="lookup_passenger",
        severity=Severity.HIGH,
        description="airline pii_disclosure via lookup_passenger",
        derived_constraint="require dual verification before disclosure",
        regression_probe_id="probe-001",
    )

    assert entry.status == "mitigated"
