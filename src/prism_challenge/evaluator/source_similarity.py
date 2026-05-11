from __future__ import annotations

import ast
import base64
import hashlib
import io
import json
import posixpath
import re
import tempfile
import tokenize
import zipfile
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SourceFile:
    path: str
    content: str
    sha256: str


@dataclass(frozen=True)
class SourceSnapshot:
    files: tuple[SourceFile, ...]
    ast_features: frozenset[str]
    token_shingles: frozenset[str]
    fingerprint: str

    @property
    def python_files(self) -> tuple[SourceFile, ...]:
        return tuple(file for file in self.files if file.path.endswith(".py"))

    def combined_python(self, *, max_chars: int = 120_000) -> str:
        chunks: list[str] = []
        used = 0
        for file in self.python_files:
            header = f"# file: {file.path}\n"
            piece = header + file.content + "\n"
            remaining = max_chars - used
            if remaining <= 0:
                break
            chunks.append(piece[:remaining])
            used += min(len(piece), remaining)
        return "\n".join(chunks)

    def to_payload(self) -> dict[str, Any]:
        return {
            "files": [
                {"path": file.path, "content": file.content, "sha256": file.sha256}
                for file in self.files
            ],
            "ast_features": sorted(self.ast_features),
            "token_shingles": sorted(self.token_shingles),
            "fingerprint": self.fingerprint,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> SourceSnapshot:
        files = tuple(
            SourceFile(
                path=str(item["path"]),
                content=str(item.get("content") or ""),
                sha256=str(item.get("sha256") or _sha256(str(item.get("content") or ""))),
            )
            for item in payload.get("files", [])
            if isinstance(item, Mapping) and item.get("path")
        )
        ast_features = frozenset(str(item) for item in payload.get("ast_features", []))
        token_shingles = frozenset(str(item) for item in payload.get("token_shingles", []))
        fingerprint = str(payload.get("fingerprint") or _fingerprint(ast_features, token_shingles))
        return cls(files, ast_features, token_shingles, fingerprint)


@dataclass(frozen=True)
class SimilarityCandidate:
    submission_id: str
    hotkey: str | None
    score: float
    ast_similarity: float
    token_similarity: float
    file_similarity: float
    snapshot: SourceSnapshot

    def summary(self) -> dict[str, Any]:
        return {
            "submission_id": self.submission_id,
            "hotkey": self.hotkey,
            "score": self.score,
            "ast_similarity": self.ast_similarity,
            "token_similarity": self.token_similarity,
            "file_similarity": self.file_similarity,
        }


SandboxRunner = Callable[[Path, Path, Path], str]
ALLOWED_PROJECT_SUFFIXES = {
    ".py",
    ".toml",
    ".yaml",
    ".yml",
    ".json",
    ".txt",
    ".md",
}


def snapshot_from_named_sources(
    sources: Iterable[tuple[str, str]],
    *,
    max_files: int = 200,
    max_bytes: int = 2_000_000,
) -> SourceSnapshot:
    extracted: dict[str, str] = {}
    total = 0
    for raw_path, content in sources:
        path = _safe_path(raw_path)
        if path.endswith(".zip"):
            for zip_path, zip_content in _extract_zip(
                path, content, max_files=max_files, max_bytes=max_bytes - total
            ).items():
                if len(extracted) >= max_files:
                    raise ValueError(f"source snapshot exceeds {max_files} files")
                total += len(zip_content.encode("utf-8"))
                if total > max_bytes:
                    raise ValueError(f"source snapshot exceeds {max_bytes} bytes")
                extracted[zip_path] = zip_content
            continue
        total += len(content.encode("utf-8"))
        if total > max_bytes:
            raise ValueError(f"source snapshot exceeds {max_bytes} bytes")
        if len(extracted) >= max_files:
            raise ValueError(f"source snapshot exceeds {max_files} files")
        extracted[path] = content
    files = tuple(
        SourceFile(path=path, content=content, sha256=_sha256(content))
        for path, content in sorted(extracted.items())
    )
    ast_features, token_shingles = _snapshot_features(files)
    return SourceSnapshot(
        files=files,
        ast_features=frozenset(ast_features),
        token_shingles=frozenset(token_shingles),
        fingerprint=_fingerprint(ast_features, token_shingles),
    )


def snapshot_from_submission(
    code: str,
    filename: str = "model.py",
    metadata: Mapping[str, Any] | None = None,
    *,
    max_files: int = 200,
    max_bytes: int = 2_000_000,
) -> SourceSnapshot:
    sources = [(filename, code)]
    archive = (metadata or {}).get("archive_base64") or (metadata or {}).get("zip_base64")
    if isinstance(archive, str) and archive.strip():
        sources.append(("metadata.zip", archive))
    return snapshot_from_named_sources(sources, max_files=max_files, max_bytes=max_bytes)


def primary_python_code(snapshot: SourceSnapshot) -> str:
    model = next((file for file in snapshot.python_files if file.path.endswith("model.py")), None)
    agent = next((file for file in snapshot.python_files if file.path.endswith("agent.py")), None)
    selected = model or agent
    if selected:
        return selected.content
    return snapshot.combined_python(max_chars=1_000_000)


def rank_similar(
    snapshot: SourceSnapshot,
    rows: Iterable[Mapping[str, Any]],
    *,
    top_k: int = 2,
    min_similarity: float = 0.65,
) -> list[SimilarityCandidate]:
    candidates: list[SimilarityCandidate] = []
    for row in rows:
        other = SourceSnapshot.from_payload(
            {
                "files": row.get("files", []),
                "ast_features": row.get("ast_features", []),
                "token_shingles": row.get("token_shingles", []),
                "fingerprint": row.get("fingerprint"),
            }
        )
        ast_similarity = jaccard(snapshot.ast_features, other.ast_features)
        token_similarity = jaccard(snapshot.token_shingles, other.token_shingles)
        file_similarity = _file_hash_similarity(snapshot, other)
        score = max(file_similarity, (0.7 * ast_similarity) + (0.3 * token_similarity))
        if score < min_similarity:
            continue
        candidates.append(
            SimilarityCandidate(
                submission_id=str(row.get("submission_id") or row.get("id") or ""),
                hotkey=str(row["hotkey"]) if row.get("hotkey") is not None else None,
                score=score,
                ast_similarity=ast_similarity,
                token_similarity=token_similarity,
                file_similarity=file_similarity,
                snapshot=other,
            )
        )
    return sorted(candidates, key=lambda item: item.score, reverse=True)[:top_k]


def build_pair_report(current: SourceSnapshot, candidate: SourceSnapshot) -> dict[str, Any]:
    current_hashes = {file.sha256: file.path for file in current.files}
    candidate_hashes = {file.sha256: file.path for file in candidate.files}
    shared_hashes = sorted(set(current_hashes) & set(candidate_hashes))
    current_paths = {file.path for file in current.files}
    candidate_paths = {file.path for file in candidate.files}
    return {
        "ast_similarity": jaccard(current.ast_features, candidate.ast_features),
        "token_similarity": jaccard(current.token_shingles, candidate.token_shingles),
        "file_similarity": _file_hash_similarity(current, candidate),
        "shared_paths": sorted(current_paths & candidate_paths),
        "exact_file_matches": [
            {"current_path": current_hashes[item], "candidate_path": candidate_hashes[item]}
            for item in shared_hashes[:50]
        ],
        "current_file_count": len(current.files),
        "candidate_file_count": len(candidate.files),
        "current_fingerprint": current.fingerprint,
        "candidate_fingerprint": candidate.fingerprint,
    }


def run_pair_sandbox(
    current: SourceSnapshot,
    candidate: SourceSnapshot,
    *,
    runner: SandboxRunner | None = None,
) -> dict[str, Any]:
    if runner is None:
        report = build_pair_report(current, candidate)
        report["sandbox"] = "local-static"
        return report
    with tempfile.TemporaryDirectory(prefix="plagiarism-pair-") as tmp:
        root = Path(tmp)
        left = root / "current"
        right = root / "candidate"
        script = root / "compare.py"
        write_snapshot_dir(current, left)
        write_snapshot_dir(candidate, right)
        script.write_text(COMPARISON_SCRIPT, encoding="utf-8")
        stdout = runner(left, right, script)
    data = json.loads(stdout)
    if not isinstance(data, dict):
        raise ValueError("pair sandbox returned non-object JSON")
    data["sandbox"] = "docker-alpine"
    return data


def write_snapshot_dir(snapshot: SourceSnapshot, root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for file in snapshot.files:
        path = root / _safe_path(file.path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(file.content, encoding="utf-8")


def jaccard(left: frozenset[str] | set[str], right: frozenset[str] | set[str]) -> float:
    if not left and not right:
        return 1.0
    return len(left & right) / max(1, len(left | right))


def _snapshot_features(files: tuple[SourceFile, ...]) -> tuple[set[str], set[str]]:
    features: set[str] = set()
    shingles: set[str] = set()
    for file in files:
        features.add(f"path:{file.path}")
        suffix = Path(file.path).suffix.lower()
        features.add(f"ext:{suffix or '<none>'}")
        tokens = _tokens(file.content)
        shingles.update(_shingles(tokens))
        if suffix != ".py":
            continue
        try:
            tree = ast.parse(file.content)
        except SyntaxError as exc:
            features.add(f"syntax_error:{exc.msg}")
            continue
        features.add(
            "ast_dump_sha256:"
            + _sha256(ast.dump(tree, annotate_fields=False, include_attributes=False))[:24]
        )
        for node in ast.walk(tree):
            features.add("node:" + type(node).__name__)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                features.add(f"function:{node.name}")
            elif isinstance(node, ast.ClassDef):
                bases = ",".join(_node_name(base) for base in node.bases)
                features.add(f"class:{node.name}:{bases}")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    features.add(f"import:{alias.name.split('.', 1)[0]}")
            elif isinstance(node, ast.ImportFrom) and node.module:
                features.add(f"import:{node.module.split('.', 1)[0]}")
            elif isinstance(node, ast.Call):
                name = _node_name(node.func)
                if name:
                    features.add(f"call:{name}")
    return features, shingles


def _tokens(content: str) -> list[str]:
    out: list[str] = []
    try:
        for token in tokenize.generate_tokens(io.StringIO(content).readline):
            if token.type in {
                tokenize.ENCODING,
                tokenize.ENDMARKER,
                tokenize.NEWLINE,
                tokenize.NL,
                tokenize.INDENT,
                tokenize.DEDENT,
                tokenize.COMMENT,
            }:
                continue
            if token.type == tokenize.STRING:
                out.append("STR")
            elif token.type == tokenize.NUMBER:
                out.append("NUM")
            else:
                out.append(token.string)
    except tokenize.TokenError:
        out.extend(re.findall(r"[A-Za-z_][A-Za-z0-9_]*|\S", content))
    return out


def _shingles(tokens: list[str], size: int = 5) -> set[str]:
    if not tokens:
        return set()
    if len(tokens) < size:
        return {" ".join(tokens)}
    return {" ".join(tokens[index : index + size]) for index in range(len(tokens) - size + 1)}


def _node_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _node_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    if isinstance(node, ast.Subscript):
        return _node_name(node.value)
    if isinstance(node, ast.Call):
        return _node_name(node.func)
    return type(node).__name__


def _file_hash_similarity(current: SourceSnapshot, candidate: SourceSnapshot) -> float:
    left = {file.sha256 for file in current.files}
    right = {file.sha256 for file in candidate.files}
    if not left and not right:
        return 1.0
    return len(left & right) / max(1, len(left | right))


def _extract_zip(path: str, content: str, *, max_files: int, max_bytes: int) -> dict[str, str]:
    try:
        raw = base64.b64decode(content, validate=True)
    except ValueError as exc:
        raise ValueError(f"zip source {path} must be base64 encoded") from exc
    out: dict[str, str] = {}
    total = 0
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            for info in archive.infolist():
                if info.is_dir():
                    continue
                if (info.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError(f"zip source {path} contains symlink: {info.filename}")
                if len(out) >= max_files:
                    raise ValueError(f"zip source {path} exceeds {max_files} files")
                nested_path = _safe_path(f"{Path(path).stem}/{info.filename}")
                suffix = Path(nested_path).suffix.lower()
                if suffix not in ALLOWED_PROJECT_SUFFIXES:
                    raise ValueError(
                        f"zip source {path} contains unsupported file: {info.filename}"
                    )
                data = archive.read(info)
                total += len(data)
                if total > max_bytes:
                    raise ValueError(f"zip source {path} exceeds {max_bytes} bytes")
                out[nested_path] = data.decode("utf-8", errors="replace")
    except zipfile.BadZipFile as exc:
        raise ValueError(f"zip source {path} is not a valid zip archive") from exc
    return out


def _safe_path(path: str) -> str:
    normalized = posixpath.normpath(path.replace("\\", "/").lstrip("/"))
    if normalized in {"", ".", ".."} or normalized.startswith("../"):
        raise ValueError(f"unsafe source path: {path}")
    parts = normalized.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"unsafe source path: {path}")
    return normalized


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()


def _fingerprint(ast_features: Iterable[str], token_shingles: Iterable[str]) -> str:
    payload = "\n".join(sorted([*ast_features, *token_shingles]))
    return _sha256(payload)


COMPARISON_SCRIPT = r"""
import hashlib
import json
import os
import re
from pathlib import Path


def collect(root: Path):
    files = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for filename in filenames:
            path = Path(dirpath) / filename
            rel = path.relative_to(root).as_posix()
            text = path.read_text(encoding="utf-8", errors="replace")
            files.append(
                {
                    "path": rel,
                    "sha256": hashlib.sha256(text.encode()).hexdigest(),
                    "tokens": tokens(text),
                }
            )
    return files


def tokens(text: str):
    return re.findall(r"[A-Za-z_][A-Za-z0-9_]*|\S", text)


def shingles(items, size=5):
    if not items:
        return set()
    if len(items) < size:
        return {" ".join(items)}
    return {" ".join(items[index:index+size]) for index in range(len(items) - size + 1)}


def jac(left, right):
    if not left and not right:
        return 1.0
    return len(left & right) / max(1, len(left | right))


left = collect(Path("/current"))
right = collect(Path("/candidate"))
left_hashes = {item["sha256"]: item["path"] for item in left}
right_hashes = {item["sha256"]: item["path"] for item in right}
left_tokens = set().union(*(shingles(item["tokens"]) for item in left)) if left else set()
right_tokens = set().union(*(shingles(item["tokens"]) for item in right)) if right else set()
print(json.dumps({
    "token_similarity": jac(left_tokens, right_tokens),
    "file_similarity": jac(set(left_hashes), set(right_hashes)),
    "shared_paths": sorted({item["path"] for item in left} & {item["path"] for item in right}),
    "exact_file_matches": [
        {"current_path": left_hashes[item], "candidate_path": right_hashes[item]}
        for item in sorted(set(left_hashes) & set(right_hashes))[:50]
    ],
    "current_file_count": len(left),
    "candidate_file_count": len(right),
}, sort_keys=True))
"""
