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


def inspect_code(code: str) -> SandboxReport:
    tree = ast.parse(code)
    imports: set[str] = set()
    fingerprint: set[str] = set()
    for node in ast.walk(tree):
        fingerprint.add(type(node).__name__)
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name.split(".", 1)[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module.split(".", 1)[0])
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in FORBIDDEN_CALLS:
                raise SandboxViolation(f"forbidden call: {node.func.id}")
    blocked = imports - ALLOWED_IMPORT_ROOTS
    if blocked:
        raise SandboxViolation(f"forbidden imports: {', '.join(sorted(blocked))}")
    required = {"build_model", "get_recipe"}
    defined = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    missing = required - defined
    if missing:
        raise SandboxViolation(f"missing functions: {', '.join(sorted(missing))}")
    return SandboxReport(tree=tree, ast_fingerprint=fingerprint, imports=imports)


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
