from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any

import yaml  # type: ignore[import-untyped]

from .source_similarity import SourceFile, SourceSnapshot, primary_python_code

MANIFEST_NAMES = {"prism.yaml", "prism.yml"}
PROJECT_KINDS = {"full", "architecture_only", "training_for_arch"}


@dataclass(frozen=True)
class PrismProjectComponents:
    kind: str
    architecture_id: str | None
    entrypoint: str
    architecture_files: tuple[SourceFile, ...]
    training_files: tuple[SourceFile, ...]
    manifest: dict[str, Any]


@dataclass(frozen=True)
class PrismComponentFingerprints:
    family_hash: str
    arch_fingerprint: str
    behavior_fingerprint: str
    training_hash: str


def project_components(snapshot: SourceSnapshot) -> PrismProjectComponents:
    manifest_file = _manifest_file(snapshot)
    if manifest_file is None:
        primary = _primary_file(snapshot)
        return PrismProjectComponents(
            kind="full",
            architecture_id=None,
            entrypoint=primary.path,
            architecture_files=(primary,),
            training_files=(primary,),
            manifest={},
        )
    manifest = yaml.safe_load(manifest_file.content) or {}
    if not isinstance(manifest, dict):
        raise ValueError("prism.yaml must contain a mapping")
    kind = str(manifest.get("kind") or "full")
    if kind not in PROJECT_KINDS:
        raise ValueError(f"unsupported Prism project kind: {kind}")
    architecture_id = manifest.get("architecture_id")
    architecture_id = str(architecture_id) if architecture_id else None
    arch_section = _section(manifest, "architecture")
    train_section = _section(manifest, "training")
    manifest_root = str(PurePosixPath(manifest_file.path).parent)
    if manifest_root == ".":
        manifest_root = ""
    arch_entrypoint = _resolve_path(
        snapshot,
        str(arch_section.get("entrypoint") or "model.py"),
        manifest_root,
    )
    train_entrypoint = _resolve_path(
        snapshot,
        str(train_section.get("entrypoint") or arch_entrypoint),
        manifest_root,
    )
    architecture_files = _component_files(snapshot, arch_section, manifest_root, arch_entrypoint)
    training_files = _component_files(snapshot, train_section, manifest_root, train_entrypoint)
    return PrismProjectComponents(
        kind=kind,
        architecture_id=architecture_id,
        entrypoint=arch_entrypoint,
        architecture_files=architecture_files,
        training_files=training_files,
        manifest=manifest,
    )


def component_fingerprints(
    components: PrismProjectComponents,
) -> PrismComponentFingerprints:
    arch_fingerprint = _files_fingerprint(components.architecture_files)
    training_hash = _files_fingerprint(components.training_files)
    return PrismComponentFingerprints(
        family_hash=arch_fingerprint,
        arch_fingerprint=arch_fingerprint,
        behavior_fingerprint=arch_fingerprint,
        training_hash=training_hash,
    )


def _manifest_file(snapshot: SourceSnapshot) -> SourceFile | None:
    return next(
        (file for file in snapshot.files if PurePosixPath(file.path).name in MANIFEST_NAMES),
        None,
    )


def _primary_file(snapshot: SourceSnapshot) -> SourceFile:
    primary = primary_python_code(snapshot)
    match = next((file for file in snapshot.python_files if file.content == primary), None)
    if match is not None:
        return match
    if snapshot.python_files:
        return snapshot.python_files[0]
    raise ValueError("Prism project contains no Python source")


def _section(manifest: dict[str, Any], name: str) -> dict[str, Any]:
    value = manifest.get(name) or {}
    if not isinstance(value, dict):
        raise ValueError(f"{name} section must be a mapping")
    return value


def _resolve_path(snapshot: SourceSnapshot, path: str, manifest_root: str) -> str:
    candidates = [path.lstrip("/")]
    if manifest_root:
        candidates.append(f"{manifest_root}/{path.lstrip('/')}")
    existing = {file.path for file in snapshot.files}
    for candidate in candidates:
        normalized = PurePosixPath(candidate).as_posix()
        if normalized in existing:
            return normalized
    for candidate in candidates:
        normalized = PurePosixPath(candidate).as_posix()
        match = next((file.path for file in snapshot.files if file.path.endswith(normalized)), None)
        if match:
            return match
    raise ValueError(f"Prism project entrypoint not found: {path}")


def _component_files(
    snapshot: SourceSnapshot,
    section: dict[str, Any],
    manifest_root: str,
    entrypoint: str,
) -> tuple[SourceFile, ...]:
    paths = {entrypoint}
    extra = section.get("files") or []
    if isinstance(extra, str):
        extra = [extra]
    if not isinstance(extra, list):
        raise ValueError("component files must be a list")
    for item in extra:
        paths.add(_resolve_path(snapshot, str(item), manifest_root))
    files = tuple(file for file in snapshot.files if file.path in paths)
    if not files:
        raise ValueError(f"Prism component has no files for entrypoint: {entrypoint}")
    return files


def _files_fingerprint(files: tuple[SourceFile, ...]) -> str:
    root = _shared_root(files)
    parts: list[str] = []
    for file in sorted(files, key=lambda item: item.path):
        content = _normalized_python(file.content) if file.path.endswith(".py") else file.content
        path = file.path.removeprefix(f"{root}/") if root else file.path
        parts.append(f"{path}\n{content}")
    return hashlib.sha256("\n---\n".join(parts).encode()).hexdigest()


def _shared_root(files: tuple[SourceFile, ...]) -> str:
    roots = {file.path.split("/", 1)[0] for file in files if "/" in file.path}
    if len(roots) != 1:
        return ""
    root = next(iter(roots))
    return root if all(file.path.startswith(f"{root}/") for file in files) else ""


def _normalized_python(content: str) -> str:
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return content
    return ast.dump(tree, annotate_fields=False, include_attributes=False)
