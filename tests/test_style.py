"""House style: no semicolons in Markdown or docstrings, no claim of scale, no statements about AI tooling."""
import ast

MARKDOWN = ["README.md", "LICENCE_CHECK.md"]
BANNED = ["at scale", "claude", "chatgpt", "copilot", "ai-assisted", "ai tool", "ai coding", "coding agent"]


def test_markdown_has_no_semicolons_or_banned_phrases(repo_root):
    for name in MARKDOWN:
        text = (repo_root / name).read_text(encoding="utf-8")
        assert ";" not in text, f"{name} has a semicolon"
        low = text.lower()
        for phrase in BANNED:
            assert phrase not in low, f"{name} says {phrase!r}"


def test_docstrings_have_no_semicolons(repo_root, pkg_name):
    for p in (repo_root / "src" / pkg_name).glob("*.py"):
        tree = ast.parse(p.read_text(encoding="utf-8"))
        nodes = [tree] + [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)]
        for n in nodes:
            doc = ast.get_docstring(n)
            if doc:
                assert ";" not in doc, f"{p.name}: the docstring of {getattr(n, 'name', 'the module')} has a semicolon"
