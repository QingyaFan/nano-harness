"""Skills: SKILL.md parsing, index selection, and two-stage loading (I6)."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from .context import estimate_text
from .messages import Message
from .tools.spec import ToolSpec

SKILL_BODY_PREFIX = "[skill loaded:"
SKILL_ALREADY_PREFIX = "[skill already loaded:"
SKILL_INDEX_HEADER = "## 可用技能"
SKILL_INDEX_HINT = (
    "用 use_skill(name) 加载正文；正文加载后可用 "
    "read_skill_resource(name, path) 读取随附文件。"
)

FM_RE = re.compile(r"\A---\n(.*?)\n---\n?", re.DOTALL)
NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")
TOKEN_RE = re.compile(r"[a-z0-9]+")


class SkillError(Exception):
    pass


class SkillParseError(SkillError):
    pass


class SkillVersionError(SkillError):
    pass


def _split_front_matter(text: str) -> tuple[dict[str, str], str]:
    match = FM_RE.match(text)
    if not match:
        raise SkillParseError("missing YAML front matter")
    meta: dict[str, str] = {}
    for line in match.group(1).splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key, value = key.strip(), value.strip()
        if not key:
            continue
        if value.startswith('"'):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                value = value.strip('"')
        elif value.startswith("'"):
            value = value.strip("'")
        meta[key] = str(value)
    return meta, text[match.end() :]


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    body: str
    version: str
    root: Path

    @property
    def tokens(self) -> int:
        return estimate_text(self.body)

    def read_resource(self, relative: str) -> str:
        if not relative or not relative.strip():
            raise SkillError("resource path must not be empty")
        if "\x00" in relative:
            raise SkillError("resource path must not contain NUL bytes")
        pure = PurePosixPath(relative)
        if pure.is_absolute():
            raise SkillError("resource path must be relative")
        if ".." in pure.parts:
            raise SkillError("resource path must stay inside the skill directory")
        root = self.root.resolve()
        target = root.joinpath(*pure.parts).resolve()
        if not target.is_relative_to(root):
            raise SkillError("resource path escapes the skill directory")
        if not target.is_file():
            raise SkillError(f"resource {relative!r} does not exist")
        return target.read_text(encoding="utf-8")


def parse_skill(path: str | Path) -> Skill:
    skill_path = Path(path)
    raw = skill_path.read_text(encoding="utf-8")
    meta, body = _split_front_matter(raw)
    name = meta.get("name", "").strip()
    description = meta.get("description", "").strip()
    if not name or not description:
        raise SkillParseError(f"{skill_path}: name and description are required")
    if not NAME_RE.match(name):
        raise SkillParseError(f"{skill_path}: invalid skill name {name!r}")
    if skill_path.parent.name != name:
        raise SkillParseError(
            f"{skill_path}: name {name!r} must match its directory "
            f"{skill_path.parent.name!r}"
        )
    if not body.strip():
        raise SkillParseError(f"{skill_path}: body must not be empty")
    version = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return Skill(
        name=name,
        description=description,
        body=body.strip(),
        version=version,
        root=skill_path.parent,
    )


@dataclass(frozen=True)
class SkillSnapshot:
    versions: dict[str, str]
    manifest: str

    def checkpoint_meta(self) -> dict[str, str]:
        return {"skills": dict(self.versions), "skill_manifest": self.manifest}


class SkillLibrary:
    def __init__(self, roots: Iterable[str | Path]) -> None:
        self._roots = [Path(root) for root in roots]
        self._skills: tuple[Skill, ...] = ()
        self._versions: dict[str, str] = {}
        self._manifest = ""
        self.scan()

    def scan(self) -> SkillSnapshot:
        found: dict[str, Skill] = {}
        for root in self._roots:
            if not root.is_dir():
                continue
            for child in sorted(root.iterdir()):
                skill_file = child / "SKILL.md"
                if not child.is_dir() or not skill_file.is_file():
                    continue
                try:
                    skill = parse_skill(skill_file)
                except SkillParseError:
                    continue
                found.setdefault(skill.name, skill)
        self._skills = tuple(sorted(found.values(), key=lambda item: item.name))
        self._versions = {skill.name: skill.version for skill in self._skills}
        material = "".join(
            f"{skill.name}\t{skill.version}\n"
            for skill in sorted(self._skills, key=lambda item: item.name)
        )
        self._manifest = hashlib.sha256(material.encode("utf-8")).hexdigest()
        return self.snapshot()

    @property
    def skills(self) -> tuple[Skill, ...]:
        return self._skills

    @property
    def versions(self) -> dict[str, str]:
        return dict(self._versions)

    @property
    def manifest(self) -> str:
        return self._manifest

    def get(self, name: str) -> Skill | None:
        for skill in self._skills:
            if skill.name == name:
                return skill
        return None

    def snapshot(self) -> SkillSnapshot:
        return SkillSnapshot(dict(self._versions), self._manifest)

    def __iter__(self) -> Iterator[Skill]:
        return iter(self._skills)

    def __len__(self) -> int:
        return len(self._skills)


def tokenize(text: str) -> set[str]:
    tokens = set(TOKEN_RE.findall(text.lower()))
    tokens |= {
        token[:-1]
        for token in tokens
        if token.endswith("s") and len(token) > 3
    }
    return tokens


def score(skill: Skill, query_tokens: set[str]) -> int:
    name_tokens = tokenize(skill.name.replace("-", " ")) | {skill.name}
    name_hits = len(name_tokens & query_tokens)
    description_hits = len(tokenize(skill.description) & query_tokens)
    return name_hits * 3 + description_hits


@dataclass(frozen=True)
class SkillIndex:
    text: str
    included: tuple[Skill, ...]
    dropped: tuple[str, ...]
    used: int
    matched: bool
    fallback: bool


class SkillSelector:
    def __init__(self, budget_tokens: int = 400) -> None:
        self._budget = budget_tokens

    def rank(
        self,
        query: str,
        skills: Sequence[Skill] | None = None,
    ) -> tuple[bool, list[Skill]]:
        if skills is None:
            raise ValueError("skills are required")
        query_tokens = tokenize(query)
        scored = []
        for skill in skills:
            value = score(skill, query_tokens)
            if value > 0:
                scored.append((value, skill))
        scored.sort(key=lambda item: (-item[0], item[1].name))
        ranked = [skill for _, skill in scored]
        if ranked:
            return True, ranked
        return False, sorted(skills, key=lambda item: item.name)

    def index(
        self,
        query: str,
        skills: Sequence[Skill],
        budget_tokens: int | None = None,
    ) -> SkillIndex:
        budget = self._budget if budget_tokens is None else budget_tokens
        matched, ranked = self.rank(query, skills)
        lines: list[str] = []
        included: list[Skill] = []
        dropped: list[str] = []
        used = 0
        for position, skill in enumerate(ranked):
            line = f"- {skill.name}: {skill.description}"
            cost = estimate_text(line + "\n")
            if used + cost > budget:
                dropped.extend(item.name for item in ranked[position:])
                break
            used += cost
            lines.append(line)
            included.append(skill)
        text = f"{SKILL_INDEX_HEADER}\n\n{SKILL_INDEX_HINT}\n\n" + "\n".join(lines)
        if lines:
            text += "\n"
        return SkillIndex(
            text=text,
            included=tuple(included),
            dropped=tuple(dropped),
            used=estimate_text(text),
            matched=matched,
            fallback=not matched,
        )


@dataclass(frozen=True)
class SkillLoad:
    skill: Skill
    reused: bool
    content: str


class SkillLoader:
    def __init__(self, library: SkillLibrary) -> None:
        self._library = library
        self._loaded: dict[str, str] = {}

    @property
    def loaded_versions(self) -> dict[str, str]:
        return dict(self._loaded)

    def load(self, name: str) -> SkillLoad:
        skill = self._library.get(name)
        if skill is None:
            raise SkillError(f"unknown skill {name!r}")
        if name in self._loaded:
            version = self._loaded[name]
            return SkillLoad(
                skill,
                True,
                f"{SKILL_ALREADY_PREFIX} {name} v{version[:8]}]",
            )
        self._loaded[name] = skill.version
        content = (
            f"{SKILL_BODY_PREFIX} {name} v{skill.version[:8]} "
            f"tokens={skill.tokens}]\n\n{skill.body}"
        )
        return SkillLoad(skill, False, content)

    def read_resource(self, name: str, relative: str) -> str:
        if name not in self._loaded:
            raise SkillError(f"skill {name!r} is not loaded")
        skill = self._library.get(name)
        if skill is None:
            raise SkillError(f"unknown skill {name!r}")
        return skill.read_resource(relative)

    def verify(self, checkpoint_versions: dict[str, str]) -> None:
        for name, version in checkpoint_versions.items():
            skill = self._library.get(name)
            if skill is None:
                raise SkillVersionError(
                    f"skill {name!r} was loaded in the checkpoint but is missing on disk"
                )
            if skill.version != version:
                raise SkillVersionError(
                    f"skill {name!r} changed on disk: checkpoint loaded "
                    f"v{version[:8]}, disk has v{skill.version[:8]}"
                )


def pin_skill_body(message: Message) -> bool:
    return message.role == "tool" and (message.content or "").startswith(
        SKILL_BODY_PREFIX
    )


def use_skill_tool(loader: SkillLoader) -> ToolSpec:
    def use(name: str) -> str:
        return loader.load(name).content

    return ToolSpec(
        name="use_skill",
        description="Load a skill body by name. Index entries are cheap; bodies are paid on use.",
        parameters={
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
            "additionalProperties": False,
        },
        handler=use,
        read_only=True,
    )


def read_skill_resource_tool(loader: SkillLoader) -> ToolSpec:
    def read(name: str, path: str) -> str:
        return loader.read_resource(name, path)

    return ToolSpec(
        name="read_skill_resource",
        description="Read a file bundled with an already loaded skill.",
        parameters={
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "path": {"type": "string"},
            },
            "required": ["name", "path"],
            "additionalProperties": False,
        },
        handler=read,
        read_only=True,
    )