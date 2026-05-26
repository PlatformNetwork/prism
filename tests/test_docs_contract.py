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


def test_readme_documents_custom_nas_submission_scope() -> None:
    readme = read_doc("README.md")

    for expected in (
        "never-before-seen architecture families",
        "configure_optimizer",
        "fallback evaluator paths may apply safe defaults or caps",
        "prism_run_manifest.v1.json",
        "FineWeb-Edu",
        "evidence-gated metric and anti-cheat review",
    ):
        assert expected in readme

    forbidden = (
        ".omo",
        "Prometheus",
        "Metis",
        "workflow artifacts",
    )
    for phrase in forbidden:
        assert phrase not in readme


def test_submission_docs_clarify_custom_training_boundaries() -> None:
    submissions_doc = read_doc("docs/submissions.md")
    miner_doc = read_doc("docs/miner/README.md")
    combined = f"{submissions_doc}\n{miner_doc}"

    for expected in (
        "It does not fix the miner architecture search space",
        "build_model(ctx)` can return any valid `torch.nn.Module`",
        "configure_optimizer` gives full optimizer and LR control",
        "fallback optimizer may apply safe evaluator defaults/caps",
        "train_step` can implement a fully custom update step",
        "custom `train_step` implementations are responsible for DDP-safe and rank-aware behavior",
        "training_for_arch` submission cannot silently change architecture family",
        "Submitted metrics are not free-form claims",
    ):
        assert expected in combined



def test_checkpoint_hook_contract_docs_are_exact() -> None:
    submissions_doc = read_doc("docs/submissions.md")
    miner_doc = read_doc("docs/miner/README.md")
    security_doc = read_doc("docs/security.md")
    combined = f"{submissions_doc}\n{miner_doc}\n{security_doc}"

    for expected in (
        "save_checkpoint(model, checkpoint_dir, ctx)",
        "load_checkpoint(model, checkpoint_dir, ctx)",
        "`checkpoint_dir`",
        "`resume_checkpoint_dir`",
        "`checkpoint_api_version`",
        "`attempt`",
        "`is_resume`",
        "`rank`",
        "`local_rank`",
        "`world_size`",
        "`distributed_backend`",
        "`device`",
        "`checkpoint_metadata`",
        "`None`, when no checkpoint artifact was produced for PRISM to record",
        "A checkpoint-dir-relative `str`",
        "should be accepted and recorded",
        'The exact shape `{"path": str, "metadata": dict[str, object]}`',
        "Writing files under `checkpoint_dir` is not enough for PRISM to record",
        "Return `None` only when no checkpoint artifact should be recorded",
        "return a checkpoint-dir-relative `str` or the exact dict shape",
        "artifact-root-relative manifest paths",
        "decimal 10G, `10_000_000_000` bytes",
        "decimal 10G, exactly `10_000_000_000` bytes",
        "retry-only after eligible infrastructure or eviction failures",
        "same submission, code, architecture, and recipe lineage",
    ):
        assert expected in combined

    forbidden_checkpoint_semantics = (
        "`None`, when the hook writes its checkpoint files directly under `checkpoint_dir`",
        "None, when the hook writes its checkpoint files directly under checkpoint_dir",
    )
    for phrase in forbidden_checkpoint_semantics:
        assert phrase not in combined


def test_distributed_docs_define_v1_single_node_scope() -> None:
    submissions_doc = read_doc("docs/submissions.md")
    miner_doc = read_doc("docs/miner/README.md")
    scaling_doc = read_doc("docs/scaling.md")
    security_doc = read_doc("docs/security.md")
    combined = f"{submissions_doc}\n{miner_doc}\n{scaling_doc}\n{security_doc}"

    for expected in (
        "single-node only",
        "Runs with 1-8 GPUs use single-node torchrun",
        "torchrun --standalone --nnodes=1 --nproc-per-node=1",
        "Requests above 8 GPUs are rejected",
        "Rank 0 writes shared checkpoint and manifest artifacts",
        "PRISM DDP-wraps default training",
        "custom `train_step` implementations are responsible for DDP-safe and rank-aware behavior",
        (
            "Custom `train_step` implementations that bypass the default loop "
            "must be DDP-safe and rank-aware"
        ),
        "does not support multi-node distributed training",
        "command and environment support, not proof that every submission succeeds on 8 GPUs",
    ):
        assert expected in combined


def test_public_docs_define_platform_gpu_broker_contract() -> None:
    scaling_doc = read_doc("docs/scaling.md")
    security_doc = read_doc("docs/security.md")
    operators_doc = read_doc("docs/operators.md")
    combined = f"{scaling_doc}\n{security_doc}\n{operators_doc}"

    for expected in (
        "`gpu_count=None` or an omitted `gpu_count` means CPU-only",
        "A positive integer requests that many GPUs",
        "Platform owns `gpu_resource_name`",
        "PRISM does not pass `gpu_resource_name`",
        "resources.limits['nvidia.com/gpu']",
        "Device IDs are not Kubernetes placement semantics",
        "device IDs remain metadata",
        "torchrun --standalone --nnodes=1 --nproc-per-node=1",
        "does not claim multi-node support",
        "Network isolation depends on the cluster CNI",
    ):
        assert expected in combined

    forbidden_claims = (
        "device IDs are Kubernetes placement semantics",
        "multi-node distributed training is supported",
        "supports arbitrary TPU",
        "supports AMD accelerator abstraction",
    )
    combined_lower = combined.lower()
    for phrase in forbidden_claims:
        assert phrase.lower() not in combined_lower


def test_checkpoint_docs_do_not_overpromise_unsupported_scope() -> None:
    public_docs = [
        "README.md",
        "docs/submissions.md",
        "docs/miner/README.md",
        "docs/scaling.md",
        "docs/security.md",
    ]
    combined = "\n".join(read_doc(path) for path in public_docs)
    combined_lower = combined.lower()

    for expected in (
        "PRISM does not support arbitrary external checkpoint resume",
        "External checkpoint paths and miner-selected resume sources are not supported",
        "Miners cannot select arbitrary external checkpoint paths or resume sources",
        "does not support object-store or cloud checkpoint upload",
        "does not support multi-node distributed training",
    ):
        assert expected in combined

    forbidden_phrases = (
        ".omo",
        "Prometheus",
        "Metis",
        "workflow artifacts",
        "arbitrary external checkpoint resume is supported",
        "miner-selected external checkpoint paths are supported",
        "object-store checkpoint upload is supported",
        "cloud checkpoint upload is supported",
        "multi-node distributed training is supported",
    )
    for phrase in forbidden_phrases:
        assert phrase.lower() not in combined_lower

def test_security_doc_clarifies_metric_review_limits() -> None:
    security_doc = read_doc("docs/security.md")

    for expected in (
        "unsupported metric or calculation claims",
        "artifacts, logs, manifests, dataset fingerprints, FLOPs, tokens, or parameter counts",
        "LLM review can inspect metric calculation claims",
        "LLM review does not replace deterministic manifest validation",
        "does not claim to recompute every metric from raw artifacts",
        "Rejection still requires deterministic evidence",
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
