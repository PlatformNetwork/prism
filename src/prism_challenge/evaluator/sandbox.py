from __future__ import annotations

import ast
import builtins
import hashlib
from dataclasses import dataclass
from types import ModuleType
from typing import Any

from .interface import PrismContext, TrainingRecipe
from .schemas import DeterministicEvidence

ALLOWED_IMPORT_ROOTS = {
    "collections",
    "dataclasses",
    "math",
    "prism_challenge",
    "torch",
    "typing",
}
NETWORK_IMPORT_ROOTS = {
    "aiohttp",
    "ftplib",
    "http",
    "httpx",
    "requests",
    "socket",
    "smtplib",
    "urllib",
}
FILESYSTEM_IMPORT_ROOTS = {"glob", "pathlib", "shutil", "tempfile"}
PROCESS_IMPORT_ROOTS = {"multiprocessing", "os", "pty", "signal", "subprocess"}
FORBIDDEN_CALL_RULES = {
    "__import__": "prism:no-forbidden-import",
    "compile": "prism:no-dynamic-code",
    "eval": "prism:no-dynamic-code",
    "exec": "prism:no-dynamic-code",
    "input": "prism:no-process",
    "open": "prism:no-filesystem",
}
FORBIDDEN_ATTRS = {
    "__builtins__",
    "__class__",
    "__dict__",
    "__globals__",
    "__import__",
    "__mro__",
    "__subclasses__",
    "system",
    "popen",
    "spawn",
    "fork",
    "remove",
    "unlink",
    "rmdir",
}
FORBIDDEN_ATTR_RULES = {
    "__builtins__": "prism:no-dynamic-code",
    "__class__": "prism:no-dynamic-code",
    "__dict__": "prism:no-dynamic-code",
    "__globals__": "prism:no-dynamic-code",
    "__import__": "prism:no-forbidden-import",
    "__mro__": "prism:no-dynamic-code",
    "__subclasses__": "prism:no-dynamic-code",
    "fork": "prism:no-process",
    "popen": "prism:no-process",
    "remove": "prism:no-filesystem",
    "rmdir": "prism:no-filesystem",
    "spawn": "prism:no-process",
    "system": "prism:no-process",
    "unlink": "prism:no-filesystem",
}
REQUIRED_CONTRACT_FUNCTIONS = {"build_model", "get_recipe"}
OPTIONAL_CONTRACT_FUNCTIONS = {
    "configure_optimizer",
    "inference_logits",
    "infer",
    "compute_loss",
    "train_step",
    "save_checkpoint",
    "load_checkpoint",
}
CONTRACT_FUNCTIONS = REQUIRED_CONTRACT_FUNCTIONS | OPTIONAL_CONTRACT_FUNCTIONS
CHECKPOINT_HOOK_SIGNATURES = {
    "save_checkpoint": ("model", "checkpoint_dir", "ctx"),
    "load_checkpoint": ("model", "checkpoint_dir", "ctx"),
}


@dataclass(frozen=True)
class SandboxReport:
    tree: ast.AST
    ast_fingerprint: set[str]
    imports: set[str]
    deterministic_evidence: tuple[DeterministicEvidence, ...] = ()


@dataclass(frozen=True)
class SubmissionRuntime:
    module: ModuleType
    model: Any
    recipe: TrainingRecipe
    report: SandboxReport
    ctx: PrismContext


class SandboxViolation(ValueError):
    def __init__(
        self,
        message: str,
        evidence: DeterministicEvidence | tuple[DeterministicEvidence, ...] | None = None,
    ) -> None:
        super().__init__(message)
        if evidence is None:
            self.evidence: tuple[DeterministicEvidence, ...] = ()
        elif isinstance(evidence, DeterministicEvidence):
            self.evidence = (evidence,)
        else:
            self.evidence = evidence

    def evidence_payload(self) -> list[dict[str, Any]]:
        return [item.model_dump() for item in self.evidence]


def inspect_code(
    code: str,
    *,
    require_contract: bool = True,
    allowed_import_roots: set[str] | None = None,
    artifact_path: str = "model.py",
) -> SandboxReport:
    tree = ast.parse(code)
    imports: set[str] = set()
    fingerprint: set[str] = set()
    for node in ast.walk(tree):
        fingerprint.add(type(node).__name__)
        if isinstance(node, ast.ClassDef):
            fingerprint.add(f"class_base:{','.join(_node_name(base) for base in node.bases)}")
        elif isinstance(node, ast.FunctionDef):
            fingerprint.add(f"function:{node.name}")
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name.split(".", 1)[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module.split(".", 1)[0])
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in FORBIDDEN_CALL_RULES:
                raise SandboxViolation(
                    f"forbidden call: {node.func.id}",
                    _evidence(
                        code,
                        node,
                        artifact_path=artifact_path,
                        rule_id=FORBIDDEN_CALL_RULES[node.func.id],
                        explanation=f"call {node.func.id} is not allowed in Prism submissions",
                    ),
                )
            fingerprint.add(f"call:{node.func.id}")
        elif isinstance(node, ast.Call):
            fingerprint.add(f"call:{_node_name(node.func)}")
        if isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_ATTRS:
            raise SandboxViolation(
                f"forbidden attribute: {node.attr}",
                _evidence(
                    code,
                    node,
                    artifact_path=artifact_path,
                    rule_id=FORBIDDEN_ATTR_RULES[node.attr],
                    explanation=f"attribute {node.attr} is not allowed in Prism submissions",
                ),
            )
    blocked = imports - ALLOWED_IMPORT_ROOTS - (allowed_import_roots or set())
    if blocked:
        evidence = tuple(
            _import_evidence(code, tree, root, artifact_path=artifact_path)
            for root in sorted(blocked)
        )
        raise SandboxViolation(f"forbidden imports: {', '.join(sorted(blocked))}", evidence)
    if require_contract:
        validate_miner_contract(tree, code=code, artifact_path=artifact_path)
    return SandboxReport(tree=tree, ast_fingerprint=fingerprint, imports=imports)


def validate_miner_contract(
    tree: ast.AST, *, code: str = "", artifact_path: str = "model.py"
) -> None:
    if not isinstance(tree, ast.Module):
        raise SandboxViolation(
            "submission must be a Python module",
            _synthetic_evidence(
                rule_id="prism:contract",
                artifact_path=artifact_path,
                ast_node=type(tree).__name__,
                basis="submission must be a Python module",
                explanation="submission must parse as a Python module",
            ),
        )
    defined = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    missing = REQUIRED_CONTRACT_FUNCTIONS - defined
    if missing:
        raise SandboxViolation(
            f"missing functions: {', '.join(sorted(missing))}",
            _synthetic_evidence(
                rule_id="prism:contract",
                artifact_path=artifact_path,
                ast_node="Module",
                basis=",".join(sorted(missing)),
                explanation="submission is missing required Prism contract functions",
            ),
        )
    allowed_top_level = (ast.ClassDef, ast.FunctionDef, ast.Import, ast.ImportFrom)
    for node in tree.body:
        if isinstance(node, allowed_top_level):
            continue
        if isinstance(node, ast.Assign | ast.AnnAssign) and _constant_assignment(node):
            continue
        raise SandboxViolation(
            "top-level code may only define imports, constants, classes, and functions",
            _evidence(
                code,
                node,
                artifact_path=artifact_path,
                rule_id="prism:no-top-level-code",
                explanation="top-level executable code is not allowed in Prism submissions",
            ),
        )
    for name in sorted(CONTRACT_FUNCTIONS & defined):
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
        if _has_varargs(fn):
            raise SandboxViolation(
                f"{name} may not use *args or **kwargs",
                _evidence(
                    code,
                    fn,
                    artifact_path=artifact_path,
                    rule_id="prism:contract",
                    explanation=f"contract function {name} may not use variable arguments",
                ),
            )
        expected_signature = CHECKPOINT_HOOK_SIGNATURES.get(name)
        if expected_signature is not None and not _has_exact_signature(
            fn, expected_signature
        ):
            raise SandboxViolation(
                f"{name} must have signature {name}({', '.join(expected_signature)})",
                _evidence(
                    code,
                    fn,
                    artifact_path=artifact_path,
                    rule_id="prism:contract",
                    explanation=(
                        f"contract function {name} must have exact parameters: "
                        f"{', '.join(expected_signature)}"
                    ),
                ),
            )


def _import_evidence(
    code: str, tree: ast.AST, root: str, *, artifact_path: str
) -> DeterministicEvidence:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name.split(".", 1)[0] == root for alias in node.names):
                return _evidence(
                    code,
                    node,
                    artifact_path=artifact_path,
                    rule_id=_import_rule_id(root),
                    explanation=f"import root {root} is not allowed in Prism submissions",
                )
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module.split(".", 1)[0] == root:
                return _evidence(
                    code,
                    node,
                    artifact_path=artifact_path,
                    rule_id=_import_rule_id(root),
                    explanation=f"import root {root} is not allowed in Prism submissions",
                )
    return _synthetic_evidence(
        rule_id=_import_rule_id(root),
        artifact_path=artifact_path,
        ast_node="Import",
        basis=root,
        explanation=f"import root {root} is not allowed in Prism submissions",
    )


def _import_rule_id(root: str) -> str:
    if root in NETWORK_IMPORT_ROOTS:
        return "prism:no-network"
    if root in FILESYSTEM_IMPORT_ROOTS:
        return "prism:no-filesystem"
    if root in PROCESS_IMPORT_ROOTS:
        return "prism:no-process"
    return "prism:no-forbidden-import"


def _evidence(
    code: str,
    node: ast.AST,
    *,
    artifact_path: str,
    rule_id: str,
    explanation: str,
) -> DeterministicEvidence:
    snippet = ast.get_source_segment(code, node) or _line_at(code, getattr(node, "lineno", 1))
    return DeterministicEvidence(
        rule_id=rule_id,
        artifact_path=artifact_path,
        line=getattr(node, "lineno", None),
        ast_node=type(node).__name__,
        snippet_hash=_sha256(snippet),
        explanation=explanation,
    )


def _synthetic_evidence(
    *, rule_id: str, artifact_path: str, ast_node: str, basis: str, explanation: str
) -> DeterministicEvidence:
    return DeterministicEvidence(
        rule_id=rule_id,
        artifact_path=artifact_path,
        ast_node=ast_node,
        snippet_hash=_sha256(basis),
        explanation=explanation,
    )


def _line_at(code: str, line: int) -> str:
    lines = code.splitlines()
    if 1 <= line <= len(lines):
        return lines[line - 1]
    return ""


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _constant_assignment(node: ast.Assign | ast.AnnAssign) -> bool:
    value = node.value
    if value is None:
        return True
    return isinstance(value, ast.Constant | ast.Tuple | ast.List | ast.Dict | ast.Set)


def _has_varargs(node: ast.FunctionDef) -> bool:
    return node.args.vararg is not None or node.args.kwarg is not None


def _has_exact_signature(node: ast.FunctionDef, expected: tuple[str, ...]) -> bool:
    args = node.args
    if args.posonlyargs or args.kwonlyargs or args.defaults or args.kw_defaults:
        return False
    return tuple(arg.arg for arg in args.args) == expected


def _node_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _node_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    if isinstance(node, ast.Subscript):
        return _node_name(node.value)
    return type(node).__name__


def _safe_import(
    name: str, globals_: Any = None, locals_: Any = None, fromlist: Any = (), level: int = 0
) -> Any:
    root = name.split(".", 1)[0]
    if root not in ALLOWED_IMPORT_ROOTS:
        raise ImportError(f"import blocked: {name}")
    return builtins.__import__(name, globals_, locals_, fromlist, level)


def load_module(code: str) -> ModuleType:
    inspect_code(code)
    module = ModuleType("prism_submission")
    safe_builtins = {
        "abs": abs,
        "all": all,
        "any": any,
        "bool": bool,
        "__build_class__": builtins.__build_class__,
        "callable": callable,
        "dict": dict,
        "enumerate": enumerate,
        "Exception": Exception,
        "float": float,
        "getattr": getattr,
        "hasattr": hasattr,
        "int": int,
        "isinstance": isinstance,
        "len": len,
        "list": list,
        "max": max,
        "min": min,
        "range": range,
        "set": set,
        "object": object,
        "super": super,
        "property": property,
        "sum": sum,
        "tuple": tuple,
        "zip": zip,
        "__import__": _safe_import,
    }
    module.__dict__["__builtins__"] = safe_builtins
    module.__dict__["__name__"] = "prism_submission"
    exec(compile(code, "<prism_submission>", "exec"), module.__dict__)
    return module


def load_submission_contract(
    code: str, ctx: PrismContext
) -> tuple[Any, TrainingRecipe, SandboxReport]:
    report = inspect_code(code)
    module = load_module(code)
    model = module.build_model(ctx)
    recipe = module.get_recipe(ctx)
    if not isinstance(recipe, TrainingRecipe):
        if isinstance(recipe, dict):
            recipe = TrainingRecipe(**recipe)
        else:
            raise SandboxViolation("get_recipe must return TrainingRecipe or dict")
    return model, recipe, report


def load_submission_runtime(code: str, ctx: PrismContext) -> SubmissionRuntime:
    report = inspect_code(code)
    module = load_module(code)
    model = module.build_model(ctx)
    recipe = module.get_recipe(ctx)
    if not isinstance(recipe, TrainingRecipe):
        if isinstance(recipe, dict):
            recipe = TrainingRecipe(**recipe)
        else:
            raise SandboxViolation("get_recipe must return TrainingRecipe or dict")
    return SubmissionRuntime(module=module, model=model, recipe=recipe, report=report, ctx=ctx)
