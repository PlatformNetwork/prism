"""Documentation contract for the PRISM v2 product.

This suite pins the public docs (README + ``docs/**``) to the ACTUAL v2 system: the two-script
submission contract (``architecture.py``/``build_model`` + ``training.py``/``train``), the locked
FineWeb-Edu data plane (read-only, no network), the forced random-init validator re-execution, the
challenge-computed prequential bits-per-byte score with a held-out delta tie-breaker, the OpenRouter
LLM hard gate, the single-node multi-GPU contract, and dry-run weights. It also asserts the docs no
longer reference the decommissioned v1-NAS machinery (component-review holds, ownership events, the
retired ``prism_run_manifest.v1.json``, or the removed ``local_cpu_smoke`` execution mode).

Assertions are anchored on real code constants so the docs cannot drift from the implementation.
"""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from prism_challenge.evaluator.interface import (
    ARCHITECTURE_FACTORY_NAME,
    DEFAULT_ARCHITECTURE_ENTRYPOINT,
    DEFAULT_TRAINING_ENTRYPOINT,
    TRAINING_ENTRYPOINT_NAME,
)
from prism_challenge.evaluator.schemas import RUN_MANIFEST_V2_FILENAME, ExecutionMode

PUBLIC_DOCS = (
    "README.md",
    "docs/overview.md",
    "docs/architecture.md",
    "docs/submissions.md",
    "docs/scoring.md",
    "docs/scaling.md",
    "docs/security.md",
    "docs/api.md",
    "docs/operators.md",
    "docs/miner/README.md",
    "docs/validator/README.md",
)


def read_doc(relative_path: str) -> str:
    return Path(relative_path).read_text(encoding="utf-8")


def all_docs_text() -> str:
    return "\n".join(read_doc(path) for path in PUBLIC_DOCS)


def test_two_script_contract_is_documented() -> None:
    readme = read_doc("README.md")
    submissions = read_doc("docs/submissions.md")
    miner = read_doc("docs/miner/README.md")
    combined = f"{readme}\n{submissions}\n{miner}"

    assert "two-script" in combined.lower()
    assert DEFAULT_ARCHITECTURE_ENTRYPOINT in combined  # architecture.py
    assert DEFAULT_TRAINING_ENTRYPOINT in combined  # training.py
    assert f"{ARCHITECTURE_FACTORY_NAME}(ctx)" in combined  # build_model(ctx)
    assert f"{TRAINING_ENTRYPOINT_NAME}(ctx)" in combined  # train(ctx)
    # The miner owns the training loop; the challenge owns the data + evaluation.
    assert "the miner owns" in combined.lower()
    submissions_lower = submissions.lower()
    for expected in (
        "the challenge owns the dataset",
        "single combined module no longer satisfies",
    ):
        assert expected in submissions_lower


def test_locked_fineweb_data_plane_is_documented() -> None:
    submissions = read_doc("docs/submissions.md")
    miner = read_doc("docs/miner/README.md")
    security = read_doc("docs/security.md")
    scaling = read_doc("docs/scaling.md")
    combined = f"{submissions}\n{miner}\n{security}\n{scaling}"

    assert "FineWeb-Edu" in combined
    assert "read-only" in combined
    assert "network=none" in combined
    assert "no network" in combined.lower()
    assert "HF_HUB_OFFLINE" in combined
    # Only the train split is miner-visible; val/test stay secret.
    assert "held-out" in combined.lower()
    assert "never exposed" in combined.lower() or "not exposed" in combined.lower()


def test_forced_init_re_execution_is_documented() -> None:
    architecture = read_doc("docs/architecture.md")
    security = read_doc("docs/security.md")
    submissions = read_doc("docs/submissions.md")
    combined = f"{architecture}\n{security}\n{submissions}"

    assert "re-execut" in combined.lower()
    assert "forced" in combined.lower()
    assert "random init" in combined.lower()
    assert "fixed seed" in combined.lower() or "forced seed" in combined.lower()
    # The challenge computes the score itself and ignores miner-reported numbers.
    assert "ignores" in combined.lower() or "ignored" in combined.lower()
    assert "miner-reported" in combined.lower()


def test_prequential_bpb_scoring_is_documented() -> None:
    scoring = read_doc("docs/scoring.md")

    assert "prequential" in scoring.lower()
    assert "bits-per-byte" in scoring.lower()
    assert "bpb" in scoring.lower()
    assert "held-out delta" in scoring.lower()
    assert "tie-break" in scoring.lower()
    assert "memorization" in scoring.lower()
    assert "tokenizer-agnostic" in scoring.lower()
    # Compute-normalized, never wall-clock.
    assert "compute-normalized" in scoring.lower()
    assert "wall-clock" in scoring.lower()
    assert "lower" in scoring.lower() and "better" in scoring.lower()


def test_llm_hard_gate_is_documented() -> None:
    security = read_doc("docs/security.md")
    operators = read_doc("docs/operators.md")
    combined = f"{security}\n{operators}"

    assert "OpenRouter" in combined
    assert "anthropic/claude-opus-4.8" in combined
    assert "hard gate" in combined.lower()
    # A reject is terminal and stops the pipeline before any GPU work.
    assert "reject" in combined.lower()
    assert "before any gpu" in combined.lower() or "before gpu" in combined.lower()


def test_multi_gpu_contract_is_documented() -> None:
    submissions = read_doc("docs/submissions.md")
    scaling = read_doc("docs/scaling.md")
    miner = read_doc("docs/miner/README.md")
    combined = f"{submissions}\n{scaling}\n{miner}"

    assert "torchrun --standalone --nnodes=1 --nproc-per-node=1" in combined
    assert "single-node" in combined.lower()
    assert "up to 8" in combined.lower() or "1-8" in combined
    assert "nproc=1" in combined
    assert "gloo" in combined.lower()
    assert "ddp" in combined.lower()


def test_dry_run_weights_are_documented() -> None:
    combined = all_docs_text()

    assert "dry-run" in combined.lower()
    assert "get_weights" in combined
    # Weights are never written on-chain.
    assert "on-chain" in combined.lower()


def test_v2_manifest_filename_is_documented() -> None:
    submissions = read_doc("docs/submissions.md")
    miner = read_doc("docs/miner/README.md")
    combined = f"{submissions}\n{miner}"

    assert RUN_MANIFEST_V2_FILENAME in combined  # prism_run_manifest.v2.json
    # The challenge authors it; any miner-written manifest is discarded.
    assert "challenge-authored" in combined.lower() or "challenge authors" in combined.lower()


def test_execution_modes_match_code() -> None:
    validator = read_doc("docs/validator/README.md")

    for mode in ExecutionMode:
        assert mode.value in validator
    # The retired local CPU smoke mode is gone from the enum and the docs.
    assert "local_cpu_smoke" not in validator


def test_docs_do_not_reference_decommissioned_machinery() -> None:
    combined = all_docs_text()
    combined_lower = combined.lower()

    forbidden = (
        "prism_run_manifest.v1",
        "component_review_holds",
        "ownership_events",
        "component_agent_reviews",
        "local_cpu_smoke",
        "run_local_cpu_smoke",
        "/internal/v1/component-review",
        "/internal/v1/worker/poll",
        "/internal/v1/validators/assignments",
        "neural architecture search",
        ".omo",
        "prometheus",
        "metis",
        "workflow artifacts",
    )
    for phrase in forbidden:
        assert phrase.lower() not in combined_lower, (
            f"decommissioned reference still present: {phrase}"
        )


def test_readme_describes_the_v2_product() -> None:
    readme = read_doc("README.md")
    readme_lower = readme.lower()

    for expected in (
        "ability to learn",
        "two-script",
        "FineWeb-Edu",
        "prequential",
        "bits-per-byte",
        "OpenRouter",
        "dry-run",
    ):
        assert expected.lower() in readme_lower, f"README missing v2 concept: {expected}"


def test_api_doc_only_lists_live_internal_routes() -> None:
    api = read_doc("docs/api.md")

    for live_route in (
        "/internal/v1/get_weights",
        "/internal/v1/bridge/submissions",
        "/internal/v1/worker/process-next",
    ):
        assert live_route in api
    for removed_route in (
        "/internal/v1/worker/poll",
        "/internal/v1/component-review/holds",
        "/internal/v1/validators/assignments",
    ):
        assert removed_route not in api


def test_architecture_lab_routes_replace_dead_nas_routes(client: TestClient) -> None:
    """The architecture-lab serving layer revives ``/v1/architectures`` and re-homes variants.

    The original v1-NAS standalone ``/v1/training-variants`` listing stayed dead (its writers were
    removed in the NAS decommission, commit ``12376e6``); training variants are now nested under
    ``/v1/architectures/{id}/variants``. ``/v1/architectures`` is a LIVE grouped leaderboard again
    (the lab producer repopulates ``architecture_families`` / ``training_variants``), so this
    contract pins the new reality: the grouped listing is served (200 with the documented shape),
    while the retired flat ``/v1/training-variants`` path remains absent (404, no route registered).
    """

    architectures = client.get("/v1/architectures")
    assert architectures.status_code == 200, architectures.text
    body = architectures.json()
    assert set(body.keys()) == {"epoch_id", "architectures"}
    assert isinstance(body["architectures"], list)

    # The old flat variant listing is gone; variants now hang off an architecture.
    assert client.get("/v1/training-variants").status_code == 404

    # The nested variant route is registered (an unknown architecture is a 404 *from the handler*,
    # never a 405/route-missing): a missing architecture yields the handler's not-found detail.
    nested = client.get("/v1/architectures/does-not-exist/variants")
    assert nested.status_code == 404, nested.text
    assert nested.json()["detail"] == "architecture not found"
