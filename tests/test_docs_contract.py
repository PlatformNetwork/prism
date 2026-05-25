from __future__ import annotations

from pathlib import Path

from prism_challenge.config import PrismSettings
from prism_challenge.runtime_config import RUNTIME_CONFIG_KEYS, runtime_policy_defaults


def read_doc(relative_path: str) -> str:
    return Path(relative_path).read_text(encoding="utf-8")


def percent_text(value: float) -> str:
    return f"{int(round(value * 100))}%"


def test_scoring_doc_matches_default_percentages() -> None:
    defaults = runtime_policy_defaults(PrismSettings())
    scoring_doc = read_doc("docs/scoring.md")

    assert "SQL runtime config can override supported policy values" in scoring_doc
    assert "Raw final loss is not a cross-architecture ranking signal" in scoring_doc
    assert "60%" in scoring_doc
    assert "40%" in scoring_doc
    assert "70% * Q_arch + 30% * Q_recipe" in scoring_doc

    score_weights = defaults["score_weights"]
    for component_name, component_weight in score_weights["architecture_formula"].items():
        assert component_name in scoring_doc
        assert percent_text(component_weight) in scoring_doc
    for component_name, component_weight in score_weights["training_formula"].items():
        assert component_name in scoring_doc
        assert percent_text(component_weight) in scoring_doc


def test_scoring_doc_lists_benchmark_weights_and_cap() -> None:
    defaults = runtime_policy_defaults(PrismSettings())
    scoring_doc = read_doc("docs/scoring.md")

    for benchmark_name, benchmark_weight in defaults["benchmark_weights"].items():
        assert benchmark_name in scoring_doc
        assert percent_text(benchmark_weight) in scoring_doc
    assert "benchmark_sanity" in scoring_doc
    assert "exceeds 15%" in scoring_doc
    assert "5% for architecture and 15% for training" in scoring_doc


def test_validator_doc_lists_sql_keys_precedence_and_audit_fields() -> None:
    validator_doc = read_doc("docs/validator/README.md")

    assert "SQL active value → env/Pydantic default → schema default" in validator_doc
    for config_key in sorted(RUNTIME_CONFIG_KEYS):
        assert f"`{config_key}`" in validator_doc
    for audit_field in (
        "config_key",
        "value_json",
        "schema_version",
        "updated_by",
        "updated_at",
        "effective_from",
        "enabled",
    ):
        assert f"`{audit_field}`" in validator_doc


def test_scaling_doc_includes_smoke_modes_and_gpu_policy() -> None:
    scaling_doc = read_doc("docs/scaling.md")

    assert "local_cpu_smoke" in scaling_doc
    assert "gpu_proxy_eval" in scaling_doc
    assert "full_scale_eval" in scaling_doc
    assert "pytest tests/test_local_cpu_smoke_eval.py -q" in scaling_doc
    assert "does not run official full-scale training" in scaling_doc
    assert "CI does not run 10B or 100B token training" in scaling_doc
    assert "`gpu_policy.max_gpu_count` | 8" in scaling_doc
    assert "Autosplit is allowed only for non-scoring" in scaling_doc


def test_security_doc_includes_evidence_gated_quarantine_policy() -> None:
    security_doc = read_doc("docs/security.md")

    assert "submit_mermaid" in security_doc
    assert "submit_verdict" in security_doc
    assert "deterministic evidence" in security_doc
    assert "quarantine" in security_doc
    assert "Suspicion-only" in security_doc or "suspicion-only" in security_doc


def test_docs_include_scientific_references_for_scoring_scaling_and_security() -> None:
    scoring_doc = read_doc("docs/scoring.md")
    scaling_doc = read_doc("docs/scaling.md")
    security_doc = read_doc("docs/security.md")

    for expected in (
        "NAS-Bench-101",
        "NAS-Bench-201",
        "Scaling Laws for Neural Language Models",
        "Training Compute-Optimal Large Language Models",
        "Green AI",
        "An Empirical Model of Large-Batch Training",
    ):
        assert expected in scoring_doc

    for expected in (
        "Broken Neural Scaling Laws",
        "The FineWeb Datasets",
        "Scaling Data-Constrained Language Models",
        "Scaling Laws and Interpretability of Learning from Repeated Data",
    ):
        assert expected in scaling_doc

    for expected in (
        "BadNets",
        "Rethinking Benchmark and Contamination",
        "Model Cards for Model Reporting",
        "Datasheets for Datasets",
    ):
        assert expected in security_doc



def test_public_docs_do_not_claim_architecture_ownership_transfer() -> None:
    public_docs = [
        "docs/scoring.md",
        "docs/validator/README.md",
        "docs/architecture.md",
        "docs/api.md",
        "docs/operators.md",
    ]
    combined = "\n".join(read_doc(path) for path in public_docs).lower()

    forbidden_phrases = (
        "architecture transfer",
        "architecture ownership transfer",
        "ownership changes and transfers",
        "architecture or training transfers",
        "architecture/training transfers",
        "architecture_action` are `new`, `existing`, `transfer`",
        "architecture actions are `new`, `existing`, `transfer`",
        "transfer-worthy improvement",
        "new/existing/transfer",
        "first-discovery owner slot unless",
    )
    for phrase in forbidden_phrases:
        assert phrase not in combined

    scoring_doc = read_doc("docs/scoring.md")
    assert "first-discovery ownership" in scoring_doc
    assert "never transfer `owner_hotkey`" in scoring_doc
    assert "first-discovery owner slot" in scoring_doc
