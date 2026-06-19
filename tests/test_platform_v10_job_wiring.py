from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from conftest import signed_headers, two_script_bundle
from fastapi.testclient import TestClient
from test_artifact_manifest import _valid_manifest

from prism_challenge.app import create_app
from prism_challenge.config import PrismSettings
from prism_challenge.evaluator.interface import TrainingRecipe
from prism_challenge.evaluator.schemas import RUN_MANIFEST_FILENAME, ExecutionMode
from prism_challenge.evaluator.training import training_recipe_fingerprint
from prism_challenge.gpu_scheduler import GpuLease
from prism_challenge.sdk.executors.docker import DockerExecutorError, DockerRunResult

# The static param-count phase (architecture.md section 4.1) instantiates build_model under the
# forced seed in the worker before any GPU work, so build_model must construct a real nn.Module.
# The actual training still runs only in the (mocked) remote Platform broker container.
REMOTE_ONLY_CODE = """
import torch

def build_model(ctx):
    return torch.nn.Linear(8, 8)

def get_recipe(ctx):
    return {'learning_rate': 0.0003, 'batch_size': 2}
"""

RETRY_CHECKPOINT_CODE = REMOTE_ONLY_CODE + """

def save_checkpoint(model, checkpoint_dir, ctx):
    return None

def load_checkpoint(model, checkpoint_dir, ctx):
    return {'loaded': True}
"""


def _submit(client: TestClient, code: str, nonce: str, metadata: dict | None = None) -> str:
    body = json.dumps(
        {
            "code": two_script_bundle(arch_code=code),
            "filename": "project.zip",
            "metadata": metadata or {},
        },
        separators=(",", ":"),
    ).encode()
    response = client.post(
        "/v1/submissions",
        content=body,
        headers={
            **signed_headers("secret", body, nonce=nonce),
            "Content-Type": "application/json",
        },
    )
    assert response.status_code == 200, response.text
    return str(response.json()["id"])


def _settings(
    db_path: Path,
    *,
    gpu_count: int = 2,
    max_gpu_count: int = 4,
    device_ids: tuple[str, ...] = ("0", "1"),
    target_gpu_count: int = 4,
) -> PrismSettings:
    return PrismSettings(
        database_url=f"sqlite+aiosqlite:///{db_path}",
        shared_token="secret",
        allow_insecure_signatures=True,
        execution_backend="platform_gpu",
        docker_enabled=True,
        docker_backend="broker",
        docker_broker_url="http://platform-docker-broker:8082",
        docker_broker_token="secret",
        platform_eval_gpu_count=gpu_count,
        platform_eval_max_gpu_count=max_gpu_count,
        platform_eval_gpu_type="l4",
        platform_gpu_targets=json.dumps(
            [{"id": "target-a", "server": "server-a", "gpu_count": target_gpu_count}],
            separators=(",", ":"),
        ),
        platform_eval_gpu_device_ids=device_ids,
        component_rewards_enabled=False,
        plagiarism_enabled=False,
        platform_eval_artifact_root=db_path.parent / "eval-artifacts",
    )


def _gpu_manifest(
    mode: ExecutionMode = ExecutionMode.GPU_PROXY_EVAL,
    *,
    gpu_count: int = 2,
    device_ids: list[str] | None = None,
) -> dict:
    manifest = _valid_manifest(mode.value)
    manifest["compute"]["gpu_count"] = gpu_count
    manifest["compute"]["gpu_type"] = "l4"
    manifest["compute"]["gpu_server"] = "server-a"
    manifest["compute"]["gpu_device_ids"] = device_ids or ["0", "1"]
    manifest["metrics"]["gpu_count"] = gpu_count
    return manifest


def _eval_job_row(db_path: Path, submission_id: str) -> sqlite3.Row:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT * FROM eval_jobs WHERE submission_id=? AND level='platform_gpu' "
            "ORDER BY attempts DESC, created_at DESC LIMIT 1",
            (submission_id,),
        ).fetchall()
    finally:
        conn.close()
    assert rows
    return rows[0]


def _eval_job_rows(db_path: Path, submission_id: str) -> list[sqlite3.Row]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        return conn.execute(
            "SELECT * FROM eval_jobs WHERE submission_id=? AND level='platform_gpu' "
            "ORDER BY attempts, created_at",
            (submission_id,),
        ).fetchall()
    finally:
        conn.close()


def _lease_rows(db_path: Path, submission_id: str) -> list[sqlite3.Row]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        return conn.execute(
            "SELECT * FROM gpu_leases WHERE submission_id=? ORDER BY created_at",
            (submission_id,),
        ).fetchall()
    finally:
        conn.close()


def _write_checkpoint_manifest(
    artifact_root: Path,
    payload: dict,
    *,
    code_hash: str | None = None,
    recipe_fingerprint: str | None = None,
):
    checkpoint_dir = (
        artifact_root / "checkpoints" / payload["submission_id"] / "attempt-1" / "current"
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "model.pt"
    checkpoint_path.write_bytes(b"checkpoint")
    checkpoint_artifact_path = checkpoint_path.relative_to(artifact_root).as_posix()
    checkpoint_dir_path = checkpoint_dir.relative_to(artifact_root).as_posix()
    metadata_path = checkpoint_dir / "checkpoint_metadata.v1.json"
    metadata = {
        "checkpoint_api_version": 1,
        "submission_id": payload["submission_id"],
        "attempt": 1,
        "code_hash": code_hash or payload["code_hash"],
        "arch_hash": payload["arch_hash"],
        "recipe_fingerprint": recipe_fingerprint
        or training_recipe_fingerprint(TrainingRecipe(learning_rate=0.0003, batch_size=2)),
        "created_at": "2026-05-25T00:00:00Z",
        "checkpoint_path": checkpoint_artifact_path,
        "hook_return": {"path": "model.pt"},
        "world_size": 1,
        "rank_writer": 0,
        "checkpoint_dir": checkpoint_dir_path,
        "bytes_total": checkpoint_path.stat().st_size,
    }
    metadata_path.write_text(json.dumps(metadata, separators=(",", ":")), encoding="utf-8")
    manifest = _gpu_manifest()
    manifest["submission_id"] = payload["submission_id"]
    manifest["compute"]["checkpoint_path"] = checkpoint_artifact_path
    manifest["artifacts"]["checkpoints"] = [
        {
            "path": checkpoint_artifact_path,
            "metadata_path": metadata_path.relative_to(artifact_root).as_posix(),
            "bytes": checkpoint_path.stat().st_size,
            "attempt": 1,
            "world_size": 1,
            "rank_writer": 0,
            "created_at": "2026-05-25T00:00:00Z",
        }
    ]
    (artifact_root / RUN_MANIFEST_FILENAME).write_text(
        json.dumps(manifest, separators=(",", ":")), encoding="utf-8"
    )


def test_gpu_job_spec_includes_platform_v10_allocation_and_artifact_contract(
    tmp_path, monkeypatch
):
    captured = {}

    def fake_run(self, spec, timeout_seconds):
        artifact_mount = next(mount for mount in spec.mounts if mount.target == "/artifacts")
        (artifact_mount.source / RUN_MANIFEST_FILENAME).write_text(
            json.dumps(_gpu_manifest(), separators=(",", ":")),
            encoding="utf-8",
        )
        captured["spec"] = spec
        captured["payload"] = json.loads((spec.mounts[0].source / "payload.json").read_text())
        captured["timeout_seconds"] = timeout_seconds
        return DockerRunResult("platform-job", "", "", 0)

    monkeypatch.setattr("prism_challenge.evaluator.container.DockerExecutor.run", fake_run)
    db_path = tmp_path / "platform-v10-job.sqlite3"
    with TestClient(create_app(_settings(db_path))) as client:
        submission_id = _submit(client, REMOTE_ONLY_CODE, nonce="gpu-job-spec")
        process = client.post(
            "/internal/v1/worker/process-next",
            headers={"Authorization": "Bearer secret"},
        )
        assert process.status_code == 200, process.text
        status = client.get(f"/v1/submissions/{submission_id}").json()

    spec = captured["spec"]
    payload = captured["payload"]
    artifact_mount = next(mount for mount in spec.mounts if mount.target == "/artifacts")
    row = _eval_job_row(db_path, submission_id)

    assert status["status"] == "completed"
    assert status["q_arch"] > 0
    assert spec.labels["platform.job"] == submission_id
    assert spec.labels["platform.task"] == "architecture"
    assert spec.limits.gpu_count == 2
    assert not hasattr(spec.limits, "gpu_resource_name")
    assert "gpu_resource_name" not in payload
    assert spec.labels["prism.actual_gpu_count"] == "2"
    assert spec.labels["prism.max_gpu_count"] == "4"
    assert spec.labels["prism.gpu_type"] == "l4"
    assert spec.labels["prism.target_id"] == "target-a"
    assert spec.labels["prism.target_server"] == "server-a"
    assert spec.labels["prism.device_ids"] == "0,1"
    assert spec.command[:4] == (
        "torchrun",
        "--standalone",
        "--nnodes=1",
        "--nproc-per-node=2",
    )
    assert spec.env["PRISM_GPU_COUNT"] == "2"
    assert spec.env["PRISM_MAX_GPU_COUNT"] == "4"
    assert spec.env["PRISM_DISTRIBUTED_BACKEND"] == "nccl"
    assert spec.env["PRISM_GPU_DEVICE_IDS"] == "0,1"
    assert spec.env["PRISM_RUN_MANIFEST_PATH"] == f"/artifacts/{RUN_MANIFEST_FILENAME}"
    assert not artifact_mount.read_only
    assert payload["gpu_allocation"] == {
        "actual_gpu_count": 2,
        "max_gpu_count": 4,
        "gpu_type": "l4",
        "target_id": "target-a",
        "target_server": "server-a",
        "device_ids": ["0", "1"],
    }
    assert spec.limits.gpu_count == int(spec.env["PRISM_GPU_COUNT"])
    assert spec.limits.gpu_count == int(spec.labels["prism.actual_gpu_count"])
    assert spec.limits.gpu_count == payload["gpu_allocation"]["actual_gpu_count"]
    assert spec.limits.gpu_count == payload["mode_spec"]["resource_profile"]["gpu_count"]
    assert payload["artifact_output"] == {
        "mount": "/artifacts",
        "path": "/artifacts",
        "manifest_path": f"/artifacts/{RUN_MANIFEST_FILENAME}",
    }
    assert payload["execution_mode"] == ExecutionMode.GPU_PROXY_EVAL.value
    assert payload["mode_spec"]["mode"] == ExecutionMode.GPU_PROXY_EVAL.value
    assert payload["mode_spec"]["command"] == list(spec.command)
    assert payload["mode_spec"]["token_budget"] == 10_000_000_000
    assert payload["mode_spec"]["dataset"]["subset"] == "sample-10BT"
    assert payload["mode_spec"]["resource_profile"]["official_fixed_profile"] is True
    assert row["status"] == "completed"
    assert row["actual_gpu_count"] == 2
    assert row["requested_gpu_count"] == 2
    assert row["target_id"] == "target-a"
    assert row["target_server"] == "server-a"
    assert json.loads(row["gpu_device_ids"]) == ["0", "1"]
    assert row["artifact_output_path"] == str(artifact_mount.source)
    assert row["run_manifest_path"] == str(artifact_mount.source / RUN_MANIFEST_FILENAME)
    assert captured["timeout_seconds"] == 900


def test_single_gpu_job_spec_uses_torchrun_with_one_process(tmp_path, monkeypatch):
    captured = {}

    def fake_run(self, spec, timeout_seconds):
        artifact_mount = next(mount for mount in spec.mounts if mount.target == "/artifacts")
        (artifact_mount.source / RUN_MANIFEST_FILENAME).write_text(
            json.dumps(_gpu_manifest(gpu_count=1, device_ids=["0"]), separators=(",", ":")),
            encoding="utf-8",
        )
        captured["spec"] = spec
        captured["payload"] = json.loads((spec.mounts[0].source / "payload.json").read_text())
        return DockerRunResult("platform-job", "", "", 0)

    monkeypatch.setattr("prism_challenge.evaluator.container.DockerExecutor.run", fake_run)
    db_path = tmp_path / "platform-v10-single-gpu-job.sqlite3"
    settings = _settings(
        db_path,
        gpu_count=1,
        max_gpu_count=1,
        device_ids=("0",),
        target_gpu_count=1,
    )
    with TestClient(create_app(settings)) as client:
        submission_id = _submit(client, REMOTE_ONLY_CODE, nonce="single-gpu-job-spec")
        process = client.post(
            "/internal/v1/worker/process-next",
            headers={"Authorization": "Bearer secret"},
        )
        assert process.status_code == 200, process.text
        status = client.get(f"/v1/submissions/{submission_id}").json()

    spec = captured["spec"]
    payload = captured["payload"]
    row = _eval_job_row(db_path, submission_id)

    assert status["status"] == "completed"
    assert spec.command[:4] == (
        "torchrun",
        "--standalone",
        "--nnodes=1",
        "--nproc-per-node=1",
    )
    assert spec.limits.gpu_count == 1
    assert not hasattr(spec.limits, "gpu_resource_name")
    assert "gpu_resource_name" not in payload
    assert spec.env["PRISM_GPU_COUNT"] == "1"
    assert "PRISM_DISTRIBUTED_BACKEND" not in spec.env
    assert spec.labels["prism.actual_gpu_count"] == "1"
    assert spec.labels["prism.max_gpu_count"] == "1"
    assert spec.labels["prism.device_ids"] == "0"
    assert payload["gpu_allocation"] == {
        "actual_gpu_count": 1,
        "max_gpu_count": 1,
        "gpu_type": "l4",
        "target_id": "target-a",
        "target_server": "server-a",
        "device_ids": ["0"],
    }
    assert spec.limits.gpu_count == int(spec.env["PRISM_GPU_COUNT"])
    assert spec.limits.gpu_count == int(spec.labels["prism.actual_gpu_count"])
    assert spec.limits.gpu_count == payload["gpu_allocation"]["actual_gpu_count"]
    assert spec.limits.gpu_count == payload["mode_spec"]["resource_profile"]["gpu_count"]
    assert row["actual_gpu_count"] == 1
    assert row["requested_gpu_count"] == 1
    assert json.loads(row["gpu_device_ids"]) == ["0"]
    assert payload["mode_spec"]["command"] == list(spec.command)


def test_mode_spec_command_uses_actual_lease_gpu_count(tmp_path, monkeypatch):
    captured = {}

    def fake_run(self, spec, timeout_seconds):
        artifact_mount = next(mount for mount in spec.mounts if mount.target == "/artifacts")
        (artifact_mount.source / RUN_MANIFEST_FILENAME).write_text(
            json.dumps(_gpu_manifest(), separators=(",", ":")),
            encoding="utf-8",
        )
        captured["spec"] = spec
        captured["payload"] = json.loads((spec.mounts[0].source / "payload.json").read_text())
        return DockerRunResult("platform-job", "", "", 0)

    monkeypatch.setattr("prism_challenge.evaluator.container.DockerExecutor.run", fake_run)
    from prism_challenge.evaluator.container import PrismContainerEvaluator
    from prism_challenge.evaluator.interface import PrismContext

    evaluator = PrismContainerEvaluator(
        settings=_settings(
            tmp_path / "lease-command.sqlite3",
            gpu_count=1,
            max_gpu_count=4,
            device_ids=("0",),
            target_gpu_count=4,
        ),
        ctx=PrismContext(sequence_length=16),
    )
    lease = GpuLease(
        id="lease-2gpu",
        submission_id="sub",
        job_id=None,
        target_id="target-a",
        target_server="server-a",
        device_ids=("0", "1"),
        gpu_count=2,
        min_gpu_count=1,
        max_gpu_count=4,
        requested_gpu_count=2,
        mode=ExecutionMode.GPU_PROXY_EVAL.value,
        tier="official",
        score_eligible=True,
        autosplit_allowed=False,
        official_fixed_profile=True,
        status="active",
        reason="test",
        created_at="2026-05-26T00:00:00Z",
    )

    evaluator.evaluate(
        submission_id="sub",
        code=REMOTE_ONLY_CODE,
        code_hash="code",
        arch_hash="arch",
        backend="platform_gpu",
        gpu_lease=lease,
    )

    assert captured["spec"].command[:4] == (
        "torchrun",
        "--standalone",
        "--nnodes=1",
        "--nproc-per-node=2",
    )
    assert captured["spec"].limits.gpu_count == 2
    assert not hasattr(captured["spec"].limits, "gpu_resource_name")
    assert "gpu_resource_name" not in captured["payload"]
    assert captured["payload"]["mode_spec"]["command"] == list(captured["spec"].command)
    assert captured["payload"]["mode_spec"]["resource_profile"]["gpu_count"] == 2
    assert captured["payload"]["mode_spec"]["resource_profile"]["gpu_device_ids"] == [
        "0",
        "1",
    ]
    assert captured["payload"]["gpu_allocation"]["actual_gpu_count"] == 2


def test_infrastructure_failure_not_submission_failure(tmp_path, monkeypatch):
    def fake_run(self, spec, timeout_seconds):
        raise DockerExecutorError("Docker broker is unavailable: connection refused")

    monkeypatch.setattr("prism_challenge.evaluator.container.DockerExecutor.run", fake_run)
    db_path = tmp_path / "platform-v10-infra.sqlite3"
    with TestClient(create_app(_settings(db_path))) as client:
        submission_id = _submit(client, REMOTE_ONLY_CODE, nonce="infra-failure")
        process = client.post(
            "/internal/v1/worker/process-next",
            headers={"Authorization": "Bearer secret"},
        )
        assert process.status_code == 200, process.text
        status = client.get(f"/v1/submissions/{submission_id}").json()

    row = _eval_job_row(db_path, submission_id)

    assert status["status"] == "pending"
    assert status["q_arch"] is None
    assert status["q_recipe"] is None
    assert "broker is unavailable" in status["error"]
    assert row["status"] == "infra_failed"
    assert row["infra_retryable"] == 1
    assert "broker is unavailable" in row["error"]


def test_infra_retry_resumes_from_validated_checkpoint_and_releases_leases(
    tmp_path, monkeypatch
):
    calls = []

    def fake_run(self, spec, timeout_seconds):
        payload = json.loads((spec.mounts[0].source / "payload.json").read_text())
        artifact_mount = next(mount for mount in spec.mounts if mount.target == "/artifacts")
        calls.append({"payload": payload, "mounts": spec.mounts, "env": spec.env})
        if len(calls) == 1:
            _write_checkpoint_manifest(artifact_mount.source, payload)
            raise DockerExecutorError("Docker broker evicted worker after checkpoint")
        assert payload["checkpoint_workspace"]["attempt"] == 2
        assert payload["context"]["resume_checkpoint_dir"] == "/resume-checkpoint"
        resume_mount = next(mount for mount in spec.mounts if mount.target == "/resume-checkpoint")
        assert resume_mount.read_only is True
        assert resume_mount.source != artifact_mount.source
        (artifact_mount.source / RUN_MANIFEST_FILENAME).write_text(
            json.dumps(_gpu_manifest(), separators=(",", ":")), encoding="utf-8"
        )
        return DockerRunResult("platform-resumed-job", "", "", 0)

    monkeypatch.setattr("prism_challenge.evaluator.container.DockerExecutor.run", fake_run)
    db_path = tmp_path / "platform-v10-retry.sqlite3"
    with TestClient(create_app(_settings(db_path))) as client:
        submission_id = _submit(client, RETRY_CHECKPOINT_CODE, nonce="infra-retry-resume")
        first = client.post(
            "/internal/v1/worker/process-next",
            headers={"Authorization": "Bearer secret"},
        )
        assert first.status_code == 200, first.text
        assert client.get(f"/v1/submissions/{submission_id}").json()["status"] == "pending"
        second = client.post(
            "/internal/v1/worker/process-next",
            headers={"Authorization": "Bearer secret"},
        )
        assert second.status_code == 200, second.text
        status = client.get(f"/v1/submissions/{submission_id}").json()

    jobs = _eval_job_rows(db_path, submission_id)
    leases = _lease_rows(db_path, submission_id)

    assert status["status"] == "completed"
    assert [job["attempts"] for job in jobs] == [1, 2]
    assert jobs[0]["status"] == "infra_failed"
    assert jobs[0]["infra_retryable"] == 1
    assert jobs[0]["artifact_output_path"]
    assert jobs[1]["status"] == "completed"
    assert calls[0]["payload"]["checkpoint_workspace"]["is_resume"] is False
    assert calls[1]["env"]["PRISM_RESUME_CHECKPOINT_DIR"] == "/resume-checkpoint"
    assert all(lease["status"] == "released" for lease in leases)


def test_retry_checkpoint_code_hash_mismatch_blocks_resume(tmp_path, monkeypatch):
    calls = []

    def fake_run(self, spec, timeout_seconds):
        payload = json.loads((spec.mounts[0].source / "payload.json").read_text())
        artifact_mount = next(mount for mount in spec.mounts if mount.target == "/artifacts")
        calls.append({"payload": payload, "mounts": spec.mounts, "env": spec.env})
        if len(calls) == 1:
            _write_checkpoint_manifest(artifact_mount.source, payload, code_hash="0" * 64)
            raise DockerExecutorError("Docker broker evicted worker after checkpoint")
        assert all(mount.target != "/resume-checkpoint" for mount in spec.mounts)
        assert "PRISM_RESUME_CHECKPOINT_DIR" not in spec.env
        (artifact_mount.source / RUN_MANIFEST_FILENAME).write_text(
            json.dumps(_gpu_manifest(), separators=(",", ":")), encoding="utf-8"
        )
        return DockerRunResult("platform-no-resume-job", "", "", 0)

    monkeypatch.setattr("prism_challenge.evaluator.container.DockerExecutor.run", fake_run)
    db_path = tmp_path / "platform-v10-retry-mismatch.sqlite3"
    with TestClient(create_app(_settings(db_path))) as client:
        submission_id = _submit(client, RETRY_CHECKPOINT_CODE, nonce="infra-retry-mismatch")
        for _ in range(2):
            process = client.post(
                "/internal/v1/worker/process-next",
                headers={"Authorization": "Bearer secret"},
            )
            assert process.status_code == 200, process.text
        status = client.get(f"/v1/submissions/{submission_id}").json()

    assert status["status"] == "completed"
    assert calls[1]["payload"]["checkpoint_workspace"]["attempt"] == 2
    assert calls[1]["payload"]["checkpoint_workspace"]["is_resume"] is False


def test_retry_checkpoint_recipe_mismatch_blocks_resume(tmp_path, monkeypatch):
    calls = []

    def fake_run(self, spec, timeout_seconds):
        payload = json.loads((spec.mounts[0].source / "payload.json").read_text())
        artifact_mount = next(mount for mount in spec.mounts if mount.target == "/artifacts")
        calls.append({"payload": payload, "mounts": spec.mounts, "env": spec.env})
        if len(calls) == 1:
            _write_checkpoint_manifest(
                artifact_mount.source,
                payload,
                recipe_fingerprint="0" * 64,
            )
            raise DockerExecutorError("Docker broker evicted worker after checkpoint")
        assert all(mount.target != "/resume-checkpoint" for mount in spec.mounts)
        assert "PRISM_RESUME_CHECKPOINT_DIR" not in spec.env
        (artifact_mount.source / RUN_MANIFEST_FILENAME).write_text(
            json.dumps(_gpu_manifest(), separators=(",", ":")), encoding="utf-8"
        )
        return DockerRunResult("platform-recipe-mismatch-job", "", "", 0)

    monkeypatch.setattr("prism_challenge.evaluator.container.DockerExecutor.run", fake_run)
    db_path = tmp_path / "platform-v10-retry-recipe-mismatch.sqlite3"
    with TestClient(create_app(_settings(db_path))) as client:
        submission_id = _submit(client, RETRY_CHECKPOINT_CODE, nonce="infra-retry-recipe")
        for _ in range(2):
            process = client.post(
                "/internal/v1/worker/process-next",
                headers={"Authorization": "Bearer secret"},
            )
            assert process.status_code == 200, process.text
        status = client.get(f"/v1/submissions/{submission_id}").json()

    assert status["status"] == "completed"
    assert calls[1]["payload"]["checkpoint_workspace"]["attempt"] == 2
    assert calls[1]["payload"]["checkpoint_workspace"]["is_resume"] is False


def test_non_infra_container_failure_with_checkpoint_does_not_resume_or_hold_lease(
    tmp_path, monkeypatch
):
    def fake_run(self, spec, timeout_seconds):
        payload = json.loads((spec.mounts[0].source / "payload.json").read_text())
        artifact_mount = next(mount for mount in spec.mounts if mount.target == "/artifacts")
        _write_checkpoint_manifest(artifact_mount.source, payload)
        return DockerRunResult("platform-failed-job", "", "miner exception", 2)

    monkeypatch.setattr("prism_challenge.evaluator.container.DockerExecutor.run", fake_run)
    db_path = tmp_path / "platform-v10-non-infra.sqlite3"
    with TestClient(create_app(_settings(db_path))) as client:
        submission_id = _submit(client, REMOTE_ONLY_CODE, nonce="non-infra-checkpoint")
        process = client.post(
            "/internal/v1/worker/process-next",
            headers={"Authorization": "Bearer secret"},
        )
        assert process.status_code == 200, process.text
        status = client.get(f"/v1/submissions/{submission_id}").json()

    row = _eval_job_row(db_path, submission_id)
    leases = _lease_rows(db_path, submission_id)

    assert status["status"] == "failed"
    assert row["status"] == "failed"
    assert row["infra_retryable"] == 0
    assert row["artifact_output_path"] is None
    assert all(lease["status"] == "released" for lease in leases)


def test_full_scale_spec_includes_frozen_dataset_resource_and_phase_targets(
    tmp_path, monkeypatch
):
    captured = {}

    def fake_run(self, spec, timeout_seconds):
        artifact_mount = next(mount for mount in spec.mounts if mount.target == "/artifacts")
        (artifact_mount.source / RUN_MANIFEST_FILENAME).write_text(
            json.dumps(_gpu_manifest(ExecutionMode.FULL_SCALE_EVAL), separators=(",", ":")),
            encoding="utf-8",
        )
        captured["payload"] = json.loads((spec.mounts[0].source / "payload.json").read_text())
        captured["spec"] = spec
        captured["timeout_seconds"] = timeout_seconds
        return DockerRunResult("platform-full-scale-job", "", "", 0)

    monkeypatch.setattr("prism_challenge.evaluator.container.DockerExecutor.run", fake_run)
    db_path = tmp_path / "platform-v10-full-scale.sqlite3"
    with TestClient(create_app(_settings(db_path))) as client:
        submission_id = _submit(
            client,
            REMOTE_ONLY_CODE,
            nonce="full-scale-spec",
            metadata={"execution_mode": ExecutionMode.FULL_SCALE_EVAL.value},
        )
        process = client.post(
            "/internal/v1/worker/process-next",
            headers={"Authorization": "Bearer secret"},
        )
        assert process.status_code == 200, process.text
        status = client.get(f"/v1/submissions/{submission_id}").json()

    payload = captured["payload"]
    mode_spec = payload["mode_spec"]
    row = _eval_job_row(db_path, submission_id)

    assert status["status"] == "completed"
    assert captured["timeout_seconds"] == 900
    assert captured["spec"].env["PRISM_EXECUTION_MODE"] == ExecutionMode.FULL_SCALE_EVAL.value
    assert captured["spec"].labels["prism.execution_mode"] == ExecutionMode.FULL_SCALE_EVAL.value
    assert payload["execution_mode"] == ExecutionMode.FULL_SCALE_EVAL.value
    assert mode_spec["mode"] == ExecutionMode.FULL_SCALE_EVAL.value
    assert mode_spec["official_score_eligible"] is True
    assert mode_spec["token_budget"] == 10_000_000_000
    assert mode_spec["parameter_target"] == 150_000_000
    assert mode_spec["dataset"]["revision"] == "fineweb-edu-contract-2026-05-25"
    assert mode_spec["dataset"]["subset"] == "sample-100BT"
    assert mode_spec["dataset"]["token_count"] == 100_000_000_000
    assert mode_spec["dataset"]["network_fallback_allowed"] is False
    assert len(mode_spec["dataset"]["train_split_fingerprint"]) == 64
    assert mode_spec["resource_profile"] == {
        "profile": "fixed_official_gpu",
        "cpus": 2.0,
        "memory": "8g",
        "gpu_count": 2,
        "max_gpu_count": 4,
        "gpu_type": "l4",
        "gpu_server": "server-a",
        "gpu_device_ids": ["0", "1"],
        "official_fixed_profile": True,
    }
    assert mode_spec["artifact_output_path"] == "/artifacts"
    assert mode_spec["run_manifest_path"] == f"/artifacts/{RUN_MANIFEST_FILENAME}"
    assert mode_spec["phases"] == [
        {
            "name": "full_scale_10b_tokens",
            "token_budget": 10_000_000_000,
            "parameter_target": 150_000_000,
            "dataset_subset": "sample-10BT",
        },
        {
            "name": "phase_2_1b_params_100b_tokens",
            "token_budget": 100_000_000_000,
            "parameter_target": 1_000_000_000,
            "dataset_subset": "sample-100BT",
        },
    ]
    assert row["status"] == "completed"
    artifact_mount = next(
        mount for mount in captured["spec"].mounts if mount.target == "/artifacts"
    )
    assert row["run_manifest_path"] == str(artifact_mount.source / RUN_MANIFEST_FILENAME)
