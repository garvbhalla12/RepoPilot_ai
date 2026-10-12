import pytest

from src.chunking.chunker import chunk_file

PY = '''"""Module doc."""
import os


def alpha():
    """First."""
    return 1


@decorator
def beta(x):
    return x + 1


class Gamma:
    def method(self):
        return "g"
'''


def test_python_chunks_follow_function_boundaries_with_correct_lines():
    chunks = chunk_file("m.py", PY, "Python")
    lines = PY.splitlines()
    by_kind = {(c.kind, c.start_line): c for c in chunks}
    assert ("function", 5) in by_kind and ("function", 10) in by_kind and ("class", 15) in by_kind
    beta = by_kind[("function", 10)]  # decorator line included
    assert beta.text.splitlines()[0] == "@decorator"
    for c in chunks:  # metadata must reproduce the text exactly
        assert c.text == "\n".join(lines[c.start_line - 1:c.end_line])
        assert c.file_path == "m.py" and c.language == "Python"


def test_long_block_split_with_overlap():
    text = "\n".join(f"line_{i} = {i}" for i in range(1, 101))
    chunks = chunk_file("data.cfg", text, "Config", max_lines=30, overlap_lines=5)
    assert chunks[0].start_line == 1 and chunks[0].end_line == 30
    assert chunks[1].start_line == 26  # 5-line overlap
    assert chunks[-1].end_line == 100
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_syntax_error_python_falls_back_without_raising():
    broken = "def oops(:\n    pass\n" + "x = 1\n" * 5
    chunks = chunk_file("bad.py", broken, "Python")
    assert chunks and chunks[0].start_line == 1


def test_empty_and_whitespace_files_yield_no_chunks():
    assert chunk_file("e.py", "", "Python") == []
    assert chunk_file("e.py", "   \n\n  ", "Python") == []


def test_markdown_and_js_boundaries():
    md = "# Title\nintro text here ok\n\n## Setup\nrun the thing now\n"
    assert [c.start_line for c in chunk_file("R.md", md, "Markdown")] == [1, 4]
    js = "const a = 1; // module constant\nexport async function load() {\n  return 2;\n}\nclass Foo {\n  x = 1;\n}\n"
    assert [c.start_line for c in chunk_file("a.js", js, "JavaScript")] == [1, 2, 5]


def test_invalid_max_lines():
    with pytest.raises(ValueError):
        chunk_file("a.py", "x = 1", "Python", max_lines=0)
