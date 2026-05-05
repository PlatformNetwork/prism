from __future__ import annotations

from dataclasses import dataclass

from prism_challenge.evaluator.lium_client import LiumClient


@dataclass
class FakeExecutor:
    id: str
    price_per_hour: float


@dataclass
class FakePod:
    id: str


class FakeLiumSdk:
    def __init__(self) -> None:
        self.removed = False
        self.command = ""
        self.uploads: list[tuple[str, str]] = []

    def ls(self, *, gpu_type: str | None, gpu_count: int) -> list[FakeExecutor]:
        assert gpu_type == "T4"
        assert gpu_count == 1
        return [FakeExecutor("expensive", 2.0), FakeExecutor("cheap", 0.5)]

    def up(
        self,
        *,
        executor_id: str,
        name: str,
        template_id: str | None,
        ports: int | None,
    ) -> dict[str, str]:
        assert executor_id == "cheap"
        assert name.startswith("prism-")
        assert template_id is None
        assert ports is None
        return {"id": "pod-1"}

    def wait_ready(self, pod_data: dict[str, str], *, timeout: int, poll_interval: int) -> FakePod:
        assert pod_data["id"] == "pod-1"
        assert timeout == 600
        assert poll_interval == 10
        return FakePod("pod-1")

    def exec(self, pod: FakePod, *, command: str) -> dict[str, object]:
        assert pod.id == "pod-1"
        self.command = command
        return {
            "success": True,
            "stdout": 'noise\nPRISM_METRICS_JSON={"q_arch":0.73,"q_recipe":0.9}\n',
            "stderr": "",
        }

    def rm(self, pod: FakePod) -> None:
        assert pod.id == "pod-1"
        self.removed = True

    def upload(self, pod: FakePod, *, local: str, remote: str) -> None:
        assert pod.id == "pod-1"
        self.uploads.append((local, remote))


async def test_lium_sdk_adapter_rents_executes_and_cleans_up() -> None:
    fake = FakeLiumSdk()
    client = LiumClient(
        base_url=None,
        token="test-token",
        backend="sdk",
        gpu_type="T4",
        sdk_factory=lambda _client: fake,
    )

    job = await client.submit_job(
        {
            "code": "def build_model(ctx): pass\ndef get_recipe(ctx): return {}",
            "context": {"vocab_size": 128, "sequence_length": 16, "max_parameters": 1000},
        },
        idempotency_key="abc123",
    )

    assert job.id == "pod-1"
    assert job.status == "completed"
    assert job.metrics == {"q_arch": 0.73, "q_recipe": 0.9}
    assert fake.removed
    assert fake.command.startswith("python3 /tmp/prism_eval_")
    assert len(fake.uploads) == 2
