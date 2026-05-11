from __future__ import annotations

import ast
import builtins
from dataclasses import dataclass
from types import ModuleType
from typing import Any

from .interface import PrismContext, TrainingRecipe

ALLOWED_IMPORT_ROOTS = {
    "collections",
    "dataclasses",
    "math",
    "prism_challenge",
    "torch",
    "typing",
}
FORBIDDEN_CALLS = {"eval", "exec", "open", "compile", "input", "__import__"}
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
REQUIRED_CONTRACT_FUNCTIONS = {"build_model", "get_recipe"}
OPTIONAL_CONTRACT_FUNCTIONS = {
    "configure_optimizer",
    "inference_logits",
    "infer",
    "compute_loss",
    "train_step",
}
CONTRACT_FUNCTIONS = REQUIRED_CONTRACT_FUNCTIONS | OPTIONAL_CONTRACT_FUNCTIONS


@dataclass(frozen=True)
class SandboxReport:
    tree: ast.AST
    ast_fingerprint: set[str]
    imports: set[str]


@dataclass(frozen=True)
class SubmissionRuntime:
    module: ModuleType
    model: Any
    recipe: TrainingRecipe
    report: SandboxReport
    ctx: PrismContext


class SandboxViolation(ValueError):
    pass


def inspect_code(
    code: str,
    *,
    require_contract: bool = True,
    allowed_import_roots: set[str] | None = None,
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
            if node.func.id in FORBIDDEN_CALLS:
                raise SandboxViolation(f"forbidden call: {node.func.id}")
            fingerprint.add(f"call:{node.func.id}")
        elif isinstance(node, ast.Call):
            fingerprint.add(f"call:{_node_name(node.func)}")
        if isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_ATTRS:
            raise SandboxViolation(f"forbidden attribute: {node.attr}")
    blocked = imports - ALLOWED_IMPORT_ROOTS - (allowed_import_roots or set())
    if blocked:
        raise SandboxViolation(f"forbidden imports: {', '.join(sorted(blocked))}")
    if require_contract:
        validate_miner_contract(tree)
    return SandboxReport(tree=tree, ast_fingerprint=fingerprint, imports=imports)


def validate_miner_contract(tree: ast.AST) -> None:
    if not isinstance(tree, ast.Module):
        raise SandboxViolation("submission must be a Python module")
    defined = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    missing = REQUIRED_CONTRACT_FUNCTIONS - defined
    if missing:
        raise SandboxViolation(f"missing functions: {', '.join(sorted(missing))}")
    allowed_top_level = (ast.ClassDef, ast.FunctionDef, ast.Import, ast.ImportFrom)
    for node in tree.body:
        if isinstance(node, allowed_top_level):
            continue
        if isinstance(node, ast.Assign | ast.AnnAssign) and _constant_assignment(node):
            continue
        raise SandboxViolation(
            "top-level code may only define imports, constants, classes, and functions"
        )
    for name in CONTRACT_FUNCTIONS & defined:
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
        if _has_varargs(fn):
            raise SandboxViolation(f"{name} may not use *args or **kwargs")


def _constant_assignment(node: ast.Assign | ast.AnnAssign) -> bool:
    value = node.value
    if value is None:
        return True
    return isinstance(value, ast.Constant | ast.Tuple | ast.List | ast.Dict | ast.Set)


def _has_varargs(node: ast.FunctionDef) -> bool:
    return node.args.vararg is not None or node.args.kwarg is not None


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
