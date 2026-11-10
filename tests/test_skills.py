import pytest

from nano_harness.context import ContextBudget, ContextManager, WindowInfo, estimate_text
from nano_harness.messages import Message, ToolCall
from nano_harness.skills import (
    SKILL_BODY_PREFIX,
    SkillError,
    SkillLibrary,
    SkillLoader,
    SkillParseError,
    SkillSelector,
    SkillVersionError,
    parse_skill,
    pin_skill_body,
    read_skill_resource_tool,
    use_skill_tool,
)
from nano_harness.tools import ToolExecutor


def write_skill(root, name, description, body, resources=None):
    directory = root / name
    directory.mkdir(parents=True)
    (directory / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n\n{body}\n",
        encoding="utf-8",
    )
    for relative, content in (resources or {}).items():
        path = directory / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return directory


def make_library(tmp_path):
    root = tmp_path / "skills"
    write_skill(
        root,
        "pdf-forms",
        "Fill and flatten PDF form fields",
        "# PDF Forms\n\nFill the fields.",
        {"references/fields.md": "FIELD LIST"},
    )
    write_skill(
        root,
        "pdf-scanned",
        "OCR a scanned PDF image",
        "# Scanned PDFs\n\nOCR the image.",
    )
    write_skill(
        root,
        "csv-report",
        "Summarize a CSV file into a Markdown report",
        "# CSV\n\nSummarize rows.",
    )
    return SkillLibrary([root])


def test_parse_skill_and_resource_paths(tmp_path) -> None:
    directory = write_skill(
        tmp_path / "skills",
        "pdf-forms",
        "Fill and flatten PDF form fields",
        "# PDF Forms\n\nBody.",
        {"references/fields.md": "FIELD LIST"},
    )
    skill = parse_skill(directory / "SKILL.md")

    assert skill.name == "pdf-forms"
    assert len(skill.version) == 64
    assert skill.tokens > 0
    assert skill.read_resource("references/fields.md") == "FIELD LIST"
    with pytest.raises(SkillError):
        skill.read_resource("../secret.txt")


def test_parse_skill_rejects_name_mismatch_and_missing_front_matter(tmp_path) -> None:
    directory = tmp_path / "wrong-name"
    directory.mkdir()
    path = directory / "SKILL.md"
    path.write_text(
        "---\nname: pdf-forms\ndescription: x\n---\n\nbody\n", encoding="utf-8"
    )
    with pytest.raises(SkillParseError):
        parse_skill(path)

    path.write_text("# no front matter\n", encoding="utf-8")
    with pytest.raises(SkillParseError):
        parse_skill(path)


def test_library_shadowing_and_manifest(tmp_path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    write_skill(first, "pdf-forms", "first root", "body one")
    write_skill(second, "pdf-forms", "second root", "body two")
    write_skill(second, "csv-report", "csv report", "body three")

    library = SkillLibrary([first, second])
    assert library.get("pdf-forms").description == "first root"
    assert set(library.versions) == {"pdf-forms", "csv-report"}
    assert len(library.manifest) == 64
    assert library.snapshot().manifest == library.manifest


def test_selector_ranking_budget_and_fallback(tmp_path) -> None:
    library = make_library(tmp_path)
    selector = SkillSelector()
    matched, ranked = selector.rank("fill pdf form", library.skills)

    assert matched is True
    assert ranked[0].name == "pdf-forms"

    index = selector.index("fill pdf form", library.skills, budget_tokens=15)
    assert index.text.startswith("## 可用技能")
    assert index.included
    assert index.dropped
    assert index.used == estimate_text(index.text)

    fallback = selector.index("zzz", library.skills, budget_tokens=1000)
    assert fallback.matched is False
    assert fallback.fallback is True
    assert {skill.name for skill in fallback.included} == {
        "pdf-forms",
        "pdf-scanned",
        "csv-report",
    }


def test_loader_two_stage_resources_and_version_drift(tmp_path) -> None:
    library = make_library(tmp_path)
    loader = SkillLoader(library)

    first = loader.load("pdf-forms")
    second = loader.load("pdf-forms")
    assert first.reused is False
    assert first.content.startswith(SKILL_BODY_PREFIX)
    assert second.reused is True
    assert loader.read_resource("pdf-forms", "references/fields.md") == "FIELD LIST"

    versions = loader.loaded_versions
    library.scan()
    loader.verify(versions)

    skill_file = tmp_path / "skills" / "pdf-forms" / "SKILL.md"
    skill_file.write_text(
        "---\nname: pdf-forms\ndescription: changed\n---\n\nnew body\n",
        encoding="utf-8",
    )
    library.scan()
    with pytest.raises(SkillVersionError):
        loader.verify(versions)

    write_skill(library._roots[0], "new-skill", "new", "new body")
    library.scan()
    loader.verify({"new-skill": library.versions["new-skill"]})


def test_context_keeps_loaded_skill_body(tmp_path) -> None:
    library = make_library(tmp_path)
    loader = SkillLoader(library)
    body = loader.load("pdf-forms").content

    budget = ContextBudget(
        WindowInfo(300, "authoritative"),
        threshold_ratio=0.5,
        target_ratio=0.5,
        reserve_ratio=0.1,
    )
    context = ContextManager(
        budget, pin=pin_skill_body, artifact_threshold=100_000
    )
    context.add(Message.system("sys"))
    context.add(Message.user("task"))
    skill_call = ToolCall("s1", "use_skill", {"name": "pdf-forms"})
    context.add(Message.assistant(None, (skill_call,)))
    context.add(Message.tool_result(skill_call, body), tool="use_skill")

    for index in range(4):
        call = ToolCall(f"c{index}", "dump", {})
        context.add(Message.assistant(None, (call,)))
        context.add(Message.tool_result(call, "row;" * 300), tool="dump")

    prepared = context.wire(())
    assert any(
        message.role == "tool" and message.content == body
        for message in prepared.messages
    )
    assert any(record.kind == "micro" for record in prepared.records)


def test_skill_tools(tmp_path) -> None:
    library = make_library(tmp_path)
    loader = SkillLoader(library)
    executor = ToolExecutor(
        [use_skill_tool(loader), read_skill_resource_tool(loader)],
        run_id="skills",
    )

    used = executor.execute(ToolCall("c1", "use_skill", {"name": "pdf-forms"}))
    assert used.ok is True
    assert used.content.startswith(SKILL_BODY_PREFIX)

    resource = executor.execute(
        ToolCall(
            "c2",
            "read_skill_resource",
            {"name": "pdf-forms", "path": "references/fields.md"},
        )
    )
    assert resource.ok is True
    assert resource.content == "FIELD LIST"