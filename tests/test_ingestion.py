import io
import stat
import zipfile

import pytest

from src.ingestion.filters import is_ignored_path, language_for, looks_binary, should_index
from src.ingestion.zip_loader import ZipValidationError, extract_repository


def make_zip(files: dict, symlink: str | None = None) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, content in files.items():
            zf.writestr(name, content)
        if symlink:
            info = zipfile.ZipInfo(symlink)
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
            zf.writestr(info, "/etc/passwd")
    return buf.getvalue()


@pytest.mark.parametrize("bad", ["../evil.py", "a/../../evil.py", "/abs/evil.py", "C:/evil.py", "..\\evil.py"])
def test_path_traversal_and_absolute_paths_rejected(tmp_path, bad):
    with pytest.raises(ZipValidationError, match="Unsafe path"):
        extract_repository(make_zip({bad: "x = 1\n", "ok.py": "y = 2\n"}), tmp_path)
    assert not list(tmp_path.rglob("evil.py"))  # nothing escaped


def test_symlink_entries_rejected(tmp_path):
    with pytest.raises(ZipValidationError, match="Symlinks"):
        extract_repository(make_zip({"a.py": "x = 1\n"}, symlink="link.py"), tmp_path)


def test_not_a_zip_rejected(tmp_path):
    with pytest.raises(ZipValidationError, match="not a valid ZIP"):
        extract_repository(b"definitely not a zip", tmp_path)


def test_limits_enforced(tmp_path):
    many = {f"f{i}.py": "x = 1\n" for i in range(10)}
    with pytest.raises(ZipValidationError, match="limit is 5"):
        extract_repository(make_zip(many), tmp_path, max_files=5)
    with pytest.raises(ZipValidationError, match="Extracted size"):
        extract_repository(make_zip({"big.py": "a" * 2_000_000}), tmp_path, max_extracted_mb=1)
    with pytest.raises(ZipValidationError, match="larger than"):
        extract_repository(make_zip({"a.py": "x = 1\n"}), tmp_path, max_zip_mb=0)


def test_empty_and_unsupported_only_archives_rejected(tmp_path):
    with pytest.raises(ZipValidationError, match="No supported source files"):
        extract_repository(make_zip({"image.png": "\x89PNG", "notes.xyz": "hello"}), tmp_path)


def test_filtering_and_common_root_stripping(tmp_path):
    files = {
        "proj/app.py": "print('hi')\n",
        "proj/src/util.ts": "export const a = 1;\n",
        "proj/node_modules/lib/index.js": "module.exports = 1;\n",
        "proj/.git/config": "[core]\n",
        "proj/.venv/lib/x.py": "x = 1\n",
        "proj/build/out.js": "var a = 1;\n",
        "proj/__pycache__/app.cpython-311.pyc": "binary",
        "proj/.env": "SECRET=abc\n",
        "proj/logo.png": "\x89PNG\x00\x01",
        "proj/data.bin.py": "\x00\x01\x02binary",
    }
    rep = extract_repository(make_zip(files), tmp_path)
    assert rep.indexed_files == ["app.py", "src/util.ts"]
    assert "node_modules/lib/index.js" in rep.skipped
    assert ".env" in rep.skipped  # secrets files never indexed
    assert rep.skipped["data.bin.py"] == "binary content"
    assert (rep.root / "src" / "util.ts").exists()


def test_uploaded_code_is_never_executed(tmp_path):
    marker = tmp_path / "pwned.txt"
    code = f"open(r'{marker.as_posix()}', 'w').write('x')\n"
    extract_repository(make_zip({"evil.py": code}), tmp_path / "repos")
    assert not marker.exists()


def test_filter_helpers():
    assert is_ignored_path("a/node_modules/b.js") and not is_ignored_path("a/src/b.js")
    assert language_for("x/Dockerfile") == "Docker" and language_for("a.PY") == "Python"
    assert language_for("server.pem") is None and language_for("app.min.js") is None
    assert looks_binary(b"ab\x00cd") and not looks_binary("héllo".encode())
    assert should_index("a.py", 10, 1, b"x = 1")[0]
    assert not should_index("a.py", 10_000_000, 512)[0]
