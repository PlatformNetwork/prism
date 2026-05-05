from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256

from .interface import PrismContext, import_torch
from .sandbox import SandboxViolation, load_submission_contract


@dataclass(frozen=True)
class L1Result:
    valid: bool
    code_hash: str
    arch_hash: str | None
    parameter_count: int
    error: str | None = None


def validate_l1(code: str, ctx: PrismContext) -> L1Result:
    code_hash = sha256(code.encode()).hexdigest()
    try:
        torch = import_torch()
        model, _recipe, report = load_submission_contract(code, ctx)
        if not isinstance(model, torch.nn.Module):
            raise SandboxViolation("build_model must return torch.nn.Module")
        param_count = sum(p.numel() for p in model.parameters())
        if param_count <= 0:
            raise SandboxViolation("model has no parameters")
        if param_count > ctx.max_parameters:
            raise SandboxViolation("model exceeds parameter limit")
        arch_basis = ":".join(sorted(report.ast_fingerprint)) + f":{param_count}"
        return L1Result(True, code_hash, sha256(arch_basis.encode()).hexdigest(), param_count)
    except Exception as exc:
        return L1Result(False, code_hash, None, 0, str(exc))
