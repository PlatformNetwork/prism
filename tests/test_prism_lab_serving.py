"""Architecture-lab serving layer: repository read methods + ``/v1`` routes + LLM auto-report.

Seeds the lab tables directly (the producer-side writers are covered in
``test_prism_lab_producer.py``) and asserts the read aggregation/ordering/404s, the curve
downsampling + compute derivation, and the non-blocking report cache/generation flow with a mocked
OpenRouter client (no real network).
"""

from __future__ import annotations

import json
from pathlib import Path

import anyio
import pytest
from fastapi.testclient import TestClient

from prism_challenge.app import create_app
from prism_challenge.config import PrismSettings
from prism_challenge.db import Database
from prism_challenge.repository import PrismRepository

NOW = "2026-06-30T12:00:00+00:00"
EARLIER = "2026-06-20T08:00:00+00:00"
EPOCH_SECONDS = 3600


# --------------------------------------------------------------------------------------------------
# Seeding helpers (work against any aiosqlite connection from ``database.connect()``).
# --------------------------------------------------------------------------------------------------
async def _insert_family(
    conn,
    *,
    architecture_id: str,
    family_hash: str,
    owner_hotkey: str,
    canonical_submission_id: str,
    q_arch_best: float,
    display_name: str | None = "Arch",
    owner_submission_id: str | None = None,
    created_at: str = EARLIER,
    updated_at: str = NOW,
) -> None:
    await conn.execute(
        "INSERT INTO architecture_families("
        "id, family_hash, arch_fingerprint, behavior_fingerprint, owner_hotkey, "
        "owner_submission_id, canonical_submission_id, q_arch_best, display_name, "
        "created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            architecture_id,
            family_hash,
            f"{family_hash}-af",
            f"{family_hash}-bf",
            owner_hotkey,
            owner_submission_id or canonical_submission_id,
            canonical_submission_id,
            q_arch_best,
            display_name,
            created_at,
            updated_at,
        ),
    )


async def _insert_submission(
    conn,
    *,
    submission_id: str,
    hotkey: str,
    epoch_id: int,
    arch_hash: str | None,
    name: str | None = None,
    status: str = "completed",
    created_at: str = NOW,
) -> None:
    await conn.execute(
        "INSERT INTO submissions("
        "id, hotkey, epoch_id, filename, code, code_hash, arch_hash, name, metadata, status, "
        "created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            submission_id,
            hotkey,
            epoch_id,
            "project.zip",
            "code",
            f"hash-{submission_id}",
            arch_hash,
            name,
            "{}",
            status,
            created_at,
            created_at,
        ),
    )


async def _insert_score(conn, *, submission_id: str, final_score: float, metrics: dict) -> None:
    await conn.execute(
        "INSERT INTO scores("
        "submission_id, q_arch, q_recipe, anti_cheat_multiplier, diversity_bonus, penalty, "
        "final_score, metrics, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (submission_id, final_score, 0.0, 1.0, 0.0, 0.0, final_score, json.dumps(metrics), NOW),
    )


async def _insert_variant(
    conn,
    *,
    variant_id: str,
    architecture_id: str,
    training_hash: str,
    owner_hotkey: str,
    submission_id: str,
    q_recipe: float,
    is_current_best: bool,
    created_at: str = NOW,
) -> None:
    await conn.execute(
        "INSERT INTO training_variants("
        "id, architecture_id, training_hash, owner_hotkey, submission_id, q_recipe, "
        "metric_mean, metric_std, is_current_best, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            variant_id,
            architecture_id,
            training_hash,
            owner_hotkey,
            submission_id,
            q_recipe,
            q_recipe,
            0.0,
            int(is_current_best),
            created_at,
            created_at,
        ),
    )


async def _insert_curve(
    conn,
    *,
    submission_id: str,
    online_loss: list[float],
    covered_bytes_cumulative: list[float],
    step0_loss: float | None,
    baseline_nats: float | None,
    compute: dict,
) -> None:
    await conn.execute(
        "INSERT INTO submission_curves("
        "submission_id, online_loss, covered_bytes_cumulative, step0_loss, baseline_nats, "
        "compute, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            submission_id,
            json.dumps(online_loss),
            json.dumps(covered_bytes_cumulative),
            step0_loss,
            baseline_nats,
            json.dumps(compute),
            NOW,
        ),
    )


async def _make_repo(tmp_path: Path) -> PrismRepository:
    database = Database(tmp_path / "lab.sqlite3")
    await database.init()
    return PrismRepository(database, EPOCH_SECONDS)


# --------------------------------------------------------------------------------------------------
# Repository method tests.
# --------------------------------------------------------------------------------------------------
async def test_list_architectures_aggregates_and_ranks(tmp_path: Path) -> None:
    repo = await _make_repo(tmp_path)
    async with repo.database.connect() as conn:
        # Family A: 2 submissions, 2 variants, best score 0.9 (epoch 100).
        await _insert_family(
            conn,
            architecture_id="af-A",
            family_hash="hashA",
            owner_hotkey="hkA",
            canonical_submission_id="subA2",
            q_arch_best=0.9,
            display_name="Alpha",
        )
        await _insert_submission(conn, submission_id="subA1", hotkey="hkA", epoch_id=100,
                                 arch_hash="hashA")
        await _insert_submission(conn, submission_id="subA2", hotkey="hkA2", epoch_id=100,
                                 arch_hash="hashA")
        await _insert_variant(conn, variant_id="tvA1", architecture_id="af-A", training_hash="tA1",
                              owner_hotkey="hkA", submission_id="subA1", q_recipe=0.5,
                              is_current_best=False)
        await _insert_variant(conn, variant_id="tvA2", architecture_id="af-A", training_hash="tA2",
                              owner_hotkey="hkA2", submission_id="subA2", q_recipe=0.9,
                              is_current_best=True)
        # Family B: 1 submission, 1 variant, best score 0.5 (epoch 100).
        await _insert_family(
            conn,
            architecture_id="af-B",
            family_hash="hashB",
            owner_hotkey="hkB",
            canonical_submission_id="subB1",
            q_arch_best=0.5,
            display_name=None,
        )
        await _insert_submission(conn, submission_id="subB1", hotkey="hkB", epoch_id=100,
                                 arch_hash="hashB")
        await _insert_variant(conn, variant_id="tvB1", architecture_id="af-B", training_hash="tB1",
                              owner_hotkey="hkB", submission_id="subB1", q_recipe=0.5,
                              is_current_best=True)

    resolved_epoch, rows = await repo.list_architectures(100)
    assert resolved_epoch == 100
    assert [r["architecture_id"] for r in rows] == ["af-A", "af-B"]
    top = rows[0]
    assert top["arch_hash"] == "hashA"
    assert top["name"] == "Alpha"
    assert top["best_final_score"] == 0.9
    assert top["best_submission_id"] == "subA2"
    assert top["variant_count"] == 2
    assert top["submission_count"] == 2
    assert rows[1]["name"] is None
    assert rows[1]["variant_count"] == 1
    assert rows[1]["submission_count"] == 1


async def test_list_architectures_epoch_filter_and_none_fallback(tmp_path: Path) -> None:
    repo = await _make_repo(tmp_path)
    async with repo.database.connect() as conn:
        await _insert_family(conn, architecture_id="af-A", family_hash="hashA", owner_hotkey="hkA",
                             canonical_submission_id="subA", q_arch_best=0.9)
        await _insert_submission(conn, submission_id="subA", hotkey="hkA", epoch_id=100,
                                 arch_hash="hashA")
        # Family C only has a submission in the LATER epoch 105.
        await _insert_family(conn, architecture_id="af-C", family_hash="hashC", owner_hotkey="hkC",
                             canonical_submission_id="subC", q_arch_best=0.7)
        await _insert_submission(conn, submission_id="subC", hotkey="hkC", epoch_id=105,
                                 arch_hash="hashC")

    # Explicit epoch scopes by submission presence.
    _, only_a = await repo.list_architectures(100)
    assert [r["architecture_id"] for r in only_a] == ["af-A"]
    _, only_c = await repo.list_architectures(105)
    assert [r["architecture_id"] for r in only_c] == ["af-C"]
    # None resolves to the most-recent non-empty epoch (105 -> family C).
    resolved, fallback_rows = await repo.list_architectures(None)
    assert resolved == 105
    assert [r["architecture_id"] for r in fallback_rows] == ["af-C"]


async def test_get_architecture_detail_and_missing(tmp_path: Path) -> None:
    repo = await _make_repo(tmp_path)
    async with repo.database.connect() as conn:
        await _insert_family(conn, architecture_id="af-A", family_hash="hashA", owner_hotkey="hkA",
                             canonical_submission_id="subA", q_arch_best=0.42, display_name="Alpha",
                             created_at=EARLIER, updated_at=NOW)
        await _insert_submission(conn, submission_id="subA", hotkey="hkA", epoch_id=100,
                                 arch_hash="hashA")
        await _insert_variant(conn, variant_id="tvA", architecture_id="af-A", training_hash="tA",
                              owner_hotkey="hkA", submission_id="subA", q_recipe=0.42,
                              is_current_best=True)

    detail = await repo.get_architecture("af-A")
    assert detail is not None
    assert detail["arch_hash"] == "hashA"
    assert detail["name"] == "Alpha"
    assert detail["best_submission_id"] == "subA"
    assert detail["best_final_score"] == 0.42
    assert detail["variant_count"] == 1
    assert detail["submission_count"] == 1
    assert detail["first_seen_at"] == EARLIER
    assert detail["updated_at"] == NOW
    assert await repo.get_architecture("missing") is None


async def test_list_training_variants_orders_best_first(tmp_path: Path) -> None:
    repo = await _make_repo(tmp_path)
    async with repo.database.connect() as conn:
        await _insert_family(conn, architecture_id="af-A", family_hash="hashA", owner_hotkey="hkA",
                             canonical_submission_id="sub3", q_arch_best=0.9)
        await _insert_variant(conn, variant_id="tv1", architecture_id="af-A", training_hash="t1",
                              owner_hotkey="hk1", submission_id="sub1", q_recipe=0.5,
                              is_current_best=False)
        await _insert_variant(conn, variant_id="tv2", architecture_id="af-A", training_hash="t2",
                              owner_hotkey="hk2", submission_id="sub2", q_recipe=0.2,
                              is_current_best=False)
        await _insert_variant(conn, variant_id="tv3", architecture_id="af-A", training_hash="t3",
                              owner_hotkey="hk3", submission_id="sub3", q_recipe=0.9,
                              is_current_best=True)

    variants = await repo.list_training_variants("af-A")
    assert [v["variant_id"] for v in variants] == ["tv3", "tv1", "tv2"]
    assert variants[0]["is_current_best"] == 1
    assert variants[0]["final_score"] == 0.9
    assert variants[0]["owner_hotkey"] == "hk3"
    assert await repo.list_training_variants("missing") == []


async def test_get_submission_curve_and_missing(tmp_path: Path) -> None:
    repo = await _make_repo(tmp_path)
    async with repo.database.connect() as conn:
        await _insert_submission(conn, submission_id="sub1", hotkey="hk", epoch_id=100,
                                 arch_hash="hashA")
        await _insert_score(conn, submission_id="sub1", final_score=0.8,
                            metrics={"prequential_bpb": 0.95, "bits_per_byte": 0.95,
                                     "tokens_consumed": 5000.0})
        await _insert_curve(conn, submission_id="sub1", online_loss=[2.5, 2.0, 1.5],
                            covered_bytes_cumulative=[100.0, 200.0, 300.0], step0_loss=2.5,
                            baseline_nats=5.6, compute={"gpu_count": 1, "model_params": 1000})

    curve = await repo.get_submission_curve("sub1")
    assert curve is not None
    assert curve["online_loss"] == [2.5, 2.0, 1.5]
    assert curve["covered_bytes_cumulative"] == [100.0, 200.0, 300.0]
    assert curve["step0_loss"] == 2.5
    assert curve["baseline_nats"] == 5.6
    assert curve["compute"] == {"gpu_count": 1, "model_params": 1000}
    assert curve["prequential_bpb"] == 0.95
    assert curve["bits_per_byte"] == 0.95
    assert curve["tokens_consumed"] == 5000.0
    assert await repo.get_submission_curve("missing") is None


async def test_architecture_report_cache_round_trip(tmp_path: Path) -> None:
    repo = await _make_repo(tmp_path)
    assert await repo.get_architecture_report("af-A") is None
    await repo.store_architecture_report(architecture_id="af-A", content="## Summary",
                                         model="test-model", source_submission_id="subA",
                                         generated_at=NOW)
    cached = await repo.get_architecture_report("af-A")
    assert cached is not None
    assert cached["content"] == "## Summary"
    assert cached["model"] == "test-model"
    assert cached["source_submission_id"] == "subA"
    assert cached["generated_at"] == NOW


# --------------------------------------------------------------------------------------------------
# Route tests (use the shared ``client`` fixture; seed via the repository connection).
# --------------------------------------------------------------------------------------------------
def _seed(client: TestClient, coro_factory) -> None:
    repository = client.app.state.repository

    async def run() -> None:
        async with repository.database.connect() as conn:
            await coro_factory(conn)

    anyio.run(run)


def test_route_list_architectures(client: TestClient) -> None:
    async def seed(conn):
        await _insert_family(conn, architecture_id="af-A", family_hash="hashA", owner_hotkey="hkA",
                             canonical_submission_id="subA", q_arch_best=0.9, display_name="Alpha")
        await _insert_submission(conn, submission_id="subA", hotkey="hkA", epoch_id=7,
                                 arch_hash="hashA")
        await _insert_variant(conn, variant_id="tvA", architecture_id="af-A", training_hash="tA",
                              owner_hotkey="hkA", submission_id="subA", q_recipe=0.9,
                              is_current_best=True)
        await _insert_family(conn, architecture_id="af-B", family_hash="hashB", owner_hotkey="hkB",
                             canonical_submission_id="subB", q_arch_best=0.4, display_name=None)
        await _insert_submission(conn, submission_id="subB", hotkey="hkB", epoch_id=7,
                                 arch_hash="hashB")

    _seed(client, seed)
    body = client.get("/v1/architectures", params={"epoch_id": 7}).json()
    assert body["epoch_id"] == 7
    assert [a["rank"] for a in body["architectures"]] == [1, 2]
    first = body["architectures"][0]
    assert first["architecture_id"] == "af-A"
    assert first["arch_hash"] == "hashA"
    assert first["name"] == "Alpha"
    assert first["best_final_score"] == 0.9
    assert first["variant_count"] == 1
    assert first["submission_count"] == 1
    # Nullable name is returned as null, never omitted.
    assert body["architectures"][1]["name"] is None
    assert "updated_at" in first


def test_route_get_architecture_and_404(client: TestClient) -> None:
    async def seed(conn):
        await _insert_family(conn, architecture_id="af-A", family_hash="hashA", owner_hotkey="hkA",
                             canonical_submission_id="subA", q_arch_best=0.9, display_name="Alpha")
        await _insert_submission(conn, submission_id="subA", hotkey="hkA", epoch_id=7,
                                 arch_hash="hashA")

    _seed(client, seed)
    ok = client.get("/v1/architectures/af-A")
    assert ok.status_code == 200, ok.text
    detail = ok.json()
    assert detail["architecture_id"] == "af-A"
    assert detail["best_submission_id"] == "subA"
    assert "first_seen_at" in detail and "updated_at" in detail
    assert client.get("/v1/architectures/missing").status_code == 404


def test_route_variants_empty_ok_and_404(client: TestClient) -> None:
    async def seed(conn):
        await _insert_family(conn, architecture_id="af-A", family_hash="hashA", owner_hotkey="hkA",
                             canonical_submission_id="subA", q_arch_best=0.9)

    _seed(client, seed)
    ok = client.get("/v1/architectures/af-A/variants")
    assert ok.status_code == 200, ok.text
    body = ok.json()
    assert body == {"architecture_id": "af-A", "variants": []}
    missing = client.get("/v1/architectures/missing/variants")
    assert missing.status_code == 404
    assert missing.json()["detail"] == "architecture not found"


def test_route_curve_downsamples_and_preserves_endpoints(client: TestClient) -> None:
    loss = [float(i) for i in range(1200)]
    cumulative = [float(i * 10) for i in range(1200)]

    async def seed(conn):
        await _insert_score(conn, submission_id="sub1", final_score=0.8,
                            metrics={"prequential_bpb": 0.9, "bits_per_byte": 0.9,
                                     "tokens_consumed": 500.0})
        await _insert_curve(conn, submission_id="sub1", online_loss=loss,
                            covered_bytes_cumulative=cumulative, step0_loss=0.0, baseline_nats=5.0,
                            compute={"gpu_count": 2, "device": "cuda:0", "model_params": 1000,
                                     "wall_clock_seconds": 720.0, "peak_vram_bytes": 123})

    _seed(client, seed)
    body = client.get("/v1/submissions/sub1/curve").json()
    series = body["loss_curve"]
    assert series["downsampled"] is True
    assert series["points"] == 500
    assert len(series["online_loss"]) == 500
    assert len(series["covered_bytes_cumulative"]) == 500
    # First and last samples are preserved for both axes.
    assert series["online_loss"][0] == 0.0
    assert series["online_loss"][-1] == 1199.0
    assert series["covered_bytes_cumulative"][0] == 0.0
    assert series["covered_bytes_cumulative"][-1] == 11990.0
    assert body["bpb"] == {"prequential_bpb": 0.9, "bits_per_byte": 0.9}
    compute = body["compute"]
    # estimated_flops = 6 * model_params * tokens_consumed; gpu_hours = gpu_count * wall / 3600.
    assert compute["estimated_flops"] == 6.0 * 1000 * 500
    assert compute["gpu_hours"] == 2 * 720.0 / 3600.0
    assert compute["tokens_consumed"] == 500
    assert compute["device"] == "cuda:0"
    assert compute["peak_vram_bytes"] == 123
    # Fields absent from the stored profile are returned as null, never omitted.
    assert compute["gpu_tier"] is None
    assert compute["peak_rss_bytes"] is None


def test_route_curve_small_series_not_downsampled(client: TestClient) -> None:
    async def seed(conn):
        await _insert_score(conn, submission_id="sub1", final_score=0.8, metrics={})
        await _insert_curve(conn, submission_id="sub1", online_loss=[2.5, 2.0, 1.5],
                            covered_bytes_cumulative=[10.0, 20.0, 30.0], step0_loss=2.5,
                            baseline_nats=None, compute={})

    _seed(client, seed)
    body = client.get("/v1/submissions/sub1/curve").json()
    series = body["loss_curve"]
    assert series["downsampled"] is False
    assert series["points"] == 3
    assert series["online_loss"] == [2.5, 2.0, 1.5]
    # No inputs -> derived compute scalars are null.
    assert body["compute"]["estimated_flops"] is None
    assert body["compute"]["gpu_hours"] is None
    assert body["compute"]["tokens_consumed"] is None


def test_route_curve_uses_stored_estimates_when_present(client: TestClient) -> None:
    async def seed(conn):
        await _insert_score(conn, submission_id="sub1", final_score=0.8,
                            metrics={"tokens_consumed": 500.0})
        await _insert_curve(conn, submission_id="sub1", online_loss=[1.0],
                            covered_bytes_cumulative=[1.0], step0_loss=1.0, baseline_nats=1.0,
                            compute={"gpu_count": 4, "model_params": 1000,
                                     "wall_clock_seconds": 36.0, "estimated_flops": 999.0,
                                     "gpu_hours": 1.5})

    _seed(client, seed)
    compute = client.get("/v1/submissions/sub1/curve").json()["compute"]
    # Pre-computed values in the stored profile are used as-is (not recomputed).
    assert compute["estimated_flops"] == 999.0
    assert compute["gpu_hours"] == 1.5


def test_route_curve_404_when_absent(client: TestClient) -> None:
    assert client.get("/v1/submissions/nope/curve").status_code == 404


def test_route_report_unavailable_without_llm_key(client: TestClient) -> None:
    # The shared fixture has no OpenRouter key configured -> generation degrades to unavailable.
    async def seed(conn):
        await _insert_family(conn, architecture_id="af-A", family_hash="hashA", owner_hotkey="hkA",
                             canonical_submission_id="subA", q_arch_best=0.9)

    _seed(client, seed)
    body = client.get("/v1/architectures/af-A/report").json()
    assert body["architecture_id"] == "af-A"
    assert body["report"]["status"] == "unavailable"
    assert body["report"]["content"] is None
    assert client.get("/v1/architectures/missing/report").status_code == 404


def test_route_report_cached_ready(client: TestClient) -> None:
    async def seed(conn):
        await _insert_family(conn, architecture_id="af-A", family_hash="hashA", owner_hotkey="hkA",
                             canonical_submission_id="subA", q_arch_best=0.9)

    _seed(client, seed)
    # A cache row whose key matches the current best submission is served as ready, even with no
    # LLM key configured (the cache hit precedes the availability check).
    repository = client.app.state.repository

    async def store() -> None:
        await repository.store_architecture_report(
            architecture_id="af-A", content="## Summary\nok", model="cached-model",
            source_submission_id="subA", generated_at=NOW,
        )

    anyio.run(store)
    report = client.get("/v1/architectures/af-A/report").json()["report"]
    assert report["status"] == "ready"
    assert report["content"] == "## Summary\nok"
    assert report["model"] == "cached-model"
    assert report["generated_at"] is not None


def test_route_report_stale_cache_is_not_served(client: TestClient) -> None:
    async def seed(conn):
        await _insert_family(conn, architecture_id="af-A", family_hash="hashA", owner_hotkey="hkA",
                             canonical_submission_id="subNEW", q_arch_best=0.9)

    _seed(client, seed)
    repository = client.app.state.repository

    async def store() -> None:
        await repository.store_architecture_report(architecture_id="af-A", content="stale",
                                                   model="m", source_submission_id="subOLD",
                                                   generated_at=NOW)

    anyio.run(store)
    # Best submission advanced to subNEW; the cache keyed to subOLD must NOT be returned ready.
    report = client.get("/v1/architectures/af-A/report").json()["report"]
    assert report["status"] != "ready"
    assert report["content"] is None


@pytest.fixture
def report_client(tmp_path: Path):
    settings = PrismSettings(
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'report.sqlite3'}",
        shared_token="secret",
        allow_insecure_signatures=True,
        fineweb_sample_count=4,
        openrouter_api_key="test-key",
        llm_review_enabled=True,
        llm_review_required=False,
        distributed_contract_policy="off",
    )
    with TestClient(create_app(settings)) as test_client:
        yield test_client


def test_route_report_generates_in_background(report_client: TestClient, monkeypatch) -> None:
    calls: list[dict] = []

    def fake_generate(facts, *, config):
        calls.append(facts)
        return "## Summary\ngenerated", "mocked-model"

    monkeypatch.setattr("prism_challenge.routes.generate_report_content", fake_generate)

    async def seed(conn):
        await _insert_family(conn, architecture_id="af-A", family_hash="hashA", owner_hotkey="hkA",
                             canonical_submission_id="subA", q_arch_best=0.9, display_name="Alpha")
        await _insert_score(conn, submission_id="subA", final_score=0.9,
                            metrics={"prequential_bpb": 0.5, "tokens_consumed": 100.0})
        await _insert_curve(conn, submission_id="subA", online_loss=[3.0, 1.0],
                            covered_bytes_cumulative=[1.0, 2.0], step0_loss=3.0, baseline_nats=5.0,
                            compute={"gpu_count": 1, "model_params": 10})

    _seed(report_client, seed)

    first = report_client.get("/v1/architectures/af-A/report").json()["report"]
    assert first["status"] == "pending"
    assert first["content"] is None
    # The background task ran (TestClient drives it to completion) and invoked the mocked client.
    assert len(calls) == 1
    assert calls[0]["name"] == "Alpha"
    assert calls[0]["prequential_bpb"] == 0.5

    second = report_client.get("/v1/architectures/af-A/report").json()["report"]
    assert second["status"] == "ready"
    assert second["content"] == "## Summary\ngenerated"
    assert second["model"] == "mocked-model"


def test_route_report_generation_error_then_unavailable(
    report_client: TestClient, monkeypatch
) -> None:
    def boom(facts, *, config):
        raise RuntimeError("openrouter exploded")

    monkeypatch.setattr("prism_challenge.routes.generate_report_content", boom)

    async def seed(conn):
        await _insert_family(conn, architecture_id="af-A", family_hash="hashA", owner_hotkey="hkA",
                             canonical_submission_id="subA", q_arch_best=0.9)

    _seed(report_client, seed)

    first = report_client.get("/v1/architectures/af-A/report").json()["report"]
    assert first["status"] == "pending"
    # The failed generation marks the architecture unavailable for the same best submission.
    second = report_client.get("/v1/architectures/af-A/report").json()["report"]
    assert second["status"] == "unavailable"
    assert second["content"] is None


# --------------------------------------------------------------------------------------------------
# Report generator module (prompt building + OpenRouter client reuse), no network.
# --------------------------------------------------------------------------------------------------
def _bare_settings(**overrides) -> PrismSettings:
    base = {
        "openrouter_api_key_file": None,
        "llm_gateway_token_file": None,
        "llm_gateway_url": None,
        "llm_gateway_token": None,
    }
    base.update(overrides)
    return PrismSettings(**base)


def test_report_generation_available_reflects_credentials() -> None:
    from prism_challenge.evaluator.architecture_report import (
        llm_report_config,
        report_generation_available,
    )

    with_key = llm_report_config(_bare_settings(openrouter_api_key="test-key"))
    assert report_generation_available(with_key) is True

    no_key = llm_report_config(_bare_settings(openrouter_api_key=None))
    assert report_generation_available(no_key) is False

    no_model = llm_report_config(_bare_settings(openrouter_api_key="k", openrouter_model=""))
    assert report_generation_available(no_model) is False


def test_build_report_prompt_grounds_only_in_facts() -> None:
    from prism_challenge.evaluator.architecture_report import build_report_prompt

    prompt = build_report_prompt(
        {
            "name": "Rotary MoE",
            "owner_hotkey": "hk1",
            "best_final_score": 0.91,
            "variant_count": 3,
            "prequential_bpb": 0.42,
            "tokens_consumed": 5000,
            "first_loss": 3.1,
            "last_loss": 1.2,
            "loss_samples": 100,
            "compute": {"model_params": 1000, "estimated_flops": 7.0, "gpu_count": 1},
        }
    )
    assert "Rotary MoE" in prompt
    assert "0.42" in prompt
    assert "3.1" in prompt and "1.2" in prompt
    # Missing facts render as 'not available' rather than fabricated values.
    empty = build_report_prompt({"compute": {}})
    assert "not available" in empty


def test_generate_report_content_uses_resolved_client(monkeypatch) -> None:
    from prism_challenge.evaluator import architecture_report as report_mod

    captured: dict = {}

    class _FakeMessage:
        content = "## Summary\nfrom fake client"

    class _FakeChat:
        def __init__(self, **kwargs):
            captured["init"] = kwargs

        def invoke(self, messages):
            captured["messages"] = messages
            return _FakeMessage()

    monkeypatch.setattr(report_mod, "_load_chat_openai", lambda: _FakeChat)
    config = report_mod.llm_report_config(_bare_settings(openrouter_api_key="test-key"))
    content, model = report_mod.generate_report_content({"name": "A", "compute": {}}, config=config)
    assert content == "## Summary\nfrom fake client"
    assert model == config.model
    # The reused client is constructed with the resolved OpenRouter endpoint + credential.
    assert captured["init"]["model"] == config.model
    assert captured["init"]["api_key"] == "test-key"
    assert captured["init"]["base_url"] == config.base_url


def test_generate_report_content_rejects_empty_completion(monkeypatch) -> None:
    from prism_challenge.evaluator import architecture_report as report_mod

    class _EmptyMessage:
        content = "   "

    class _FakeChat:
        def __init__(self, **kwargs):
            pass

        def invoke(self, messages):
            return _EmptyMessage()

    monkeypatch.setattr(report_mod, "_load_chat_openai", lambda: _FakeChat)
    config = report_mod.llm_report_config(_bare_settings(openrouter_api_key="test-key"))
    with pytest.raises(RuntimeError):
        report_mod.generate_report_content({"compute": {}}, config=config)
