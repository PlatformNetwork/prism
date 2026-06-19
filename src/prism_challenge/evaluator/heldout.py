"""Host-side held-out delta + anti-memorization gap (architecture.md sections 5, 6).

The held-out delta-over-random-init is the scoring TIE-BREAKER:
``delta = bpb(random-init twin on val) - bpb(trained model on val)`` where a LARGER improvement
ranks better. The challenge also records the train-vs-held-out gap
(``gap = bpb(trained on val) - prequential train bpb``) as an ANTI-MEMORIZATION flag and penalizes
an excessive gap.

The secret ``val`` split is NEVER bind-mounted into the network=none eval container (VAL-HARNESS-015
/ VAL-CHEAT-007) and is never exposed via ``PrismContext``; only the CHALLENGE SCORER reads it, here
on the worker host. To compute the held-out bpb the in-container runner persists the trained
weights (``trained_state.pt``); this module rebuilds the miner architecture under the FORCED seed
(both a random-init twin and a fresh model loaded with the trained weights) and evaluates both on
val. The factory + forward run in a short-lived, time/memory-bounded ``forkserver`` child (mirroring
``static_instantiation``) so a hostile model cannot stall or balloon the worker.
"""

from __future__ import annotations

import importlib.util
import math
import multiprocessing as mp
import os
import resource
import sys
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .dataset import LockedDatasetError, iter_locked_documents
from .interface import PrismContext
from .scoring import MEMORIZATION_GAP_THRESHOLD_BPB

DEFAULT_TIMEOUT_SECONDS = 120.0
DEFAULT_MEMORY_HEADROOM_BYTES = 8 * 1024**3
DEFAULT_HELDOUT_BATCH_SIZE = 8
# A NaN/Inf held-out batch loss is sanitized to a worst-case (high) per-batch code-length so a
# degenerate model can never collapse into a finite, advantageous held-out bpb.
WORST_CASE_LOSS_MULTIPLIER = 2.0


@dataclass(frozen=True)
class HeldoutResult:
    """Challenge-computed held-out delta + anti-memorization gap on the SECRET val split."""

    val_bpb_trained: float
    val_bpb_random_init: float
    heldout_delta: float
    train_heldout_gap: float | None
    memorization_flag: bool

    def as_metrics(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "val_bpb_trained": self.val_bpb_trained,
            "val_bpb_random_init": self.val_bpb_random_init,
            "heldout_delta": self.heldout_delta,
            "held_out_delta": self.heldout_delta,
            "memorization_flag": self.memorization_flag,
        }
        if self.train_heldout_gap is not None:
            payload["train_heldout_gap"] = self.train_heldout_gap
            payload["train_val_gap"] = self.train_heldout_gap
        return payload


def _build_context() -> Any:
    try:
        context = mp.get_context("forkserver")
        context.set_forkserver_preload(["torch"])
        return context
    except (ValueError, OSError):  # pragma: no cover - platform without forkserver
        return mp.get_context("fork")


_MP_CONTEXT = _build_context()


def val_split_present(val_data_dir: str | os.PathLike[str] | None) -> bool:
    """True when a non-empty ``val`` split is resolvable at ``val_data_dir`` (host-only)."""
    if not val_data_dir:
        return False
    base = Path(val_data_dir)
    if not base.is_dir():
        return False
    if any(base.glob("val-*.jsonl")):
        return True
    nested = base / "val"
    return nested.is_dir() and any(nested.glob("val-*.jsonl"))


def compute_heldout_metrics(
    *,
    files: Mapping[str, str],
    entrypoint: str,
    ctx: PrismContext,
    trained_state_path: str | os.PathLike[str],
    val_data_dir: str | os.PathLike[str] | None,
    train_bpb: float | None,
    build_model_symbol: str = "build_model",
    batch_size: int = DEFAULT_HELDOUT_BATCH_SIZE,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    memory_headroom_bytes: int = DEFAULT_MEMORY_HEADROOM_BYTES,
    gap_threshold_bpb: float = MEMORIZATION_GAP_THRESHOLD_BPB,
) -> HeldoutResult | None:
    """Compute the held-out delta + anti-memorization gap, or ``None`` to skip gracefully.

    Returns ``None`` (held-out skipped, the run still scores on prequential bpb) when the trained
    weights are absent, the secret val split is unavailable, or the bounded child cannot produce a
    finite/positive held-out bpb for both models.
    """
    state_path = Path(trained_state_path)
    if not state_path.is_file():
        return None
    if not val_split_present(val_data_dir):
        return None

    parent_conn, child_conn = _MP_CONTEXT.Pipe(duplex=False)
    proc = _MP_CONTEXT.Process(
        target=_child_heldout,
        args=(
            dict(files),
            entrypoint,
            build_model_symbol,
            ctx,
            str(state_path),
            str(val_data_dir),
            int(batch_size),
            int(memory_headroom_bytes),
            child_conn,
        ),
    )
    proc.start()
    child_conn.close()
    result: tuple[str, object] | None = None
    try:
        if parent_conn.poll(timeout_seconds):
            try:
                result = parent_conn.recv()
            except EOFError:
                result = None
    finally:
        if proc.is_alive():
            proc.terminate()
            proc.join(5)
            if proc.is_alive():
                proc.kill()
                proc.join()
        else:
            proc.join()
        parent_conn.close()

    if result is None:
        return None
    kind, detail = result
    if kind != "ok" or not isinstance(detail, Mapping):
        return None
    val_bpb_random = detail.get("val_bpb_random_init")
    val_bpb_trained = detail.get("val_bpb_trained")
    if not _is_sane_bpb(val_bpb_random) or not _is_sane_bpb(val_bpb_trained):
        return None
    val_bpb_random = float(val_bpb_random)  # type: ignore[arg-type]
    val_bpb_trained = float(val_bpb_trained)  # type: ignore[arg-type]
    heldout_delta = val_bpb_random - val_bpb_trained
    gap: float | None = None
    flag = False
    if train_bpb is not None and math.isfinite(float(train_bpb)):
        gap = val_bpb_trained - float(train_bpb)
        flag = gap > gap_threshold_bpb
    return HeldoutResult(
        val_bpb_trained=val_bpb_trained,
        val_bpb_random_init=val_bpb_random,
        heldout_delta=heldout_delta,
        train_heldout_gap=gap,
        memorization_flag=flag,
    )


def _is_sane_bpb(value: object) -> bool:
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) > 0.0
    )


def _child_heldout(
    files: Mapping[str, str],
    entrypoint: str,
    symbol: str,
    ctx: PrismContext,
    trained_state_path: str,
    val_data_dir: str,
    batch_size: int,
    memory_headroom_bytes: int,
    conn: Any,
) -> None:
    try:
        import torch

        torch.set_num_threads(1)
        if memory_headroom_bytes > 0:
            try:
                cap = _vmsize_bytes() + memory_headroom_bytes
                resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
            except (ValueError, OSError):
                pass

        workdir = tempfile.mkdtemp(prefix="prism-heldout-")
        for rel_path, content in files.items():
            dest = os.path.join(workdir, rel_path)
            parent = os.path.dirname(dest)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(dest, "w", encoding="utf-8") as handle:
                handle.write(content)
        entry_path = os.path.join(workdir, entrypoint)
        sys.path.insert(0, os.path.dirname(entry_path))
        sys.path.insert(0, workdir)

        spec = importlib.util.spec_from_file_location("prism_heldout_arch", entry_path)
        if spec is None or spec.loader is None:
            conn.send(("error", f"cannot load architecture entrypoint {entrypoint}"))
            return
        module = importlib.util.module_from_spec(spec)
        sys.modules["prism_heldout_arch"] = module
        spec.loader.exec_module(module)
        factory = getattr(module, symbol, None)
        if factory is None or not callable(factory):
            conn.send(("error", f"architecture is missing callable {symbol}"))
            return

        texts = _load_val_texts(val_data_dir)
        if not texts:
            conn.send(("skip", "val split empty"))
            return

        vocab_size = max(int(ctx.vocab_size), 2)
        seq_len = max(int(ctx.sequence_length), 2)
        baseline_nats = math.log(vocab_size)

        # Random-init twin: forced-seed instantiation reproduces the trained model's step-0 weights.
        torch.manual_seed(int(ctx.seed))
        twin = factory(ctx)
        if not isinstance(twin, torch.nn.Module):
            conn.send(("error", "build_model returned a non-nn.Module value"))
            return
        val_bpb_random = _bpb_over_texts(
            twin, texts, vocab_size=vocab_size, seq_len=seq_len,
            batch_size=batch_size, baseline_nats=baseline_nats,
        )

        # Trained model: a fresh instance loaded with the persisted trained weights.
        torch.manual_seed(int(ctx.seed))
        trained = factory(ctx)
        if not isinstance(trained, torch.nn.Module):
            conn.send(("error", "build_model returned a non-nn.Module value"))
            return
        state = torch.load(trained_state_path, map_location="cpu")
        trained.load_state_dict(_strip_module_prefix(state), strict=False)
        val_bpb_trained = _bpb_over_texts(
            trained, texts, vocab_size=vocab_size, seq_len=seq_len,
            batch_size=batch_size, baseline_nats=baseline_nats,
        )

        conn.send((
            "ok",
            {"val_bpb_random_init": val_bpb_random, "val_bpb_trained": val_bpb_trained},
        ))
    except BaseException as exc:  # noqa: BLE001 - report any failure cleanly to the parent
        try:
            conn.send(("error", f"{type(exc).__name__}: {exc}"))
        except Exception:
            pass
    finally:
        try:
            conn.close()
        except Exception:
            pass
        os._exit(0)


def _load_val_texts(val_data_dir: str) -> list[str]:
    try:
        return [doc.text for doc in iter_locked_documents(val_data_dir, "val")]
    except LockedDatasetError:
        return []


def _strip_module_prefix(state: Any) -> Any:
    if not isinstance(state, Mapping):
        return state
    prefix = "module."
    if not any(isinstance(key, str) and key.startswith(prefix) for key in state):
        return dict(state)
    stripped: dict[Any, Any] = {}
    for key, value in state.items():
        new_key = key[len(prefix) :] if isinstance(key, str) and key.startswith(prefix) else key
        stripped[new_key] = value
    return stripped


def _bpb_over_texts(
    model: Any,
    texts: list[str],
    *,
    vocab_size: int,
    seq_len: int,
    batch_size: int,
    baseline_nats: float,
) -> float | None:
    """Single-pass byte-level bits-per-byte of a FIXED model over the held-out texts (no training).

    The denominator is raw UTF-8 BYTES covered (tokenizer-agnostic, mirroring the prequential train
    metric); byte-level tokenization matches the runner's default instrument. Returns ``None`` for
    an empty / non-finite measurement so it never collapses into a fabricated score.
    """
    import torch
    import torch.nn.functional as functional

    vocab = max(int(vocab_size), 2)
    seq = max(int(seq_len), 2)
    bs = max(int(batch_size), 1)
    needed = seq * bs
    worst = baseline_nats * WORST_CASE_LOSS_MULTIPLIER
    try:
        model_device = next(model.parameters()).device
    except StopIteration:
        model_device = torch.device("cpu")
    was_training = model.training
    model.eval()
    sum_nll_nats = 0.0
    covered_bytes = 0
    samples = 0

    def _flush(chunk: list[int]) -> None:
        nonlocal sum_nll_nats, covered_bytes, samples
        count = len(chunk)
        if count < 2:
            return
        if count == needed:
            rows, cols = bs, seq
        else:
            rows, cols = 1, count
        tokens = (
            torch.tensor(chunk, dtype=torch.long).remainder(vocab).view(rows, cols).to(model_device)
        )
        with torch.no_grad():
            logits = _logits_tensor(model(tokens))
        if logits.dim() == 2:
            logits = logits.unsqueeze(0)
        predictions = logits[:, :-1, :].reshape(-1, vocab)
        targets = (tokens[:, 1:].reshape(-1)) % vocab
        loss = float(functional.cross_entropy(predictions, targets).detach().item())
        if not math.isfinite(loss):
            loss = worst
        predicted = rows * max(cols - 1, 0)
        sum_nll_nats += loss * predicted
        covered_bytes += count
        samples += 1

    buffer: list[int] = []
    for text in texts:
        for byte in text.encode("utf-8"):
            buffer.append(int(byte))
            if len(buffer) >= needed:
                _flush(buffer[:needed])
                buffer = buffer[needed:]
    if buffer:
        _flush(buffer)

    if was_training:
        model.train()
    if covered_bytes <= 0 or samples == 0:
        return None
    bits = sum_nll_nats / math.log(2.0)
    if not math.isfinite(bits):
        return None
    bpb = bits / covered_bytes
    return bpb if math.isfinite(bpb) and bpb > 0.0 else None


def _logits_tensor(logits: Any) -> Any:
    import torch

    if isinstance(logits, torch.Tensor):
        return logits
    if isinstance(logits, tuple | list) and logits and isinstance(logits[0], torch.Tensor):
        return logits[0]
    if hasattr(logits, "logits"):
        return logits.logits
    raise RuntimeError("miner model forward did not return a logits tensor")


def _vmsize_bytes() -> int:
    try:
        with open("/proc/self/status", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmSize:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        return 0
    return 0
