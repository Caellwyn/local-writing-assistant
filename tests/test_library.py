"""Tests for the sandbox: path safety, read-only sources, limits, history."""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import library  # noqa: E402
from library import Library, LibraryError, clean_title, safe_path  # noqa: E402


@pytest.fixture
def lib(tmp_path):
    return Library(tmp_path / "lib")


@pytest.fixture
def sample(tmp_path):
    f = tmp_path / "chapter one.txt"
    f.write_text("The lighthouse keeper counted ships.\n\nThe fog came in at dusk.", encoding="utf-8")
    return f


# -- safe_path ---------------------------------------------------------------

ESCAPES = [
    r"..\outside.txt",
    r"..\..\..\Windows\evil.txt",
    r"Drafts\..\..\evil.txt",
    "../outside.txt",
    r"C:\Windows\evil.txt",
    "C:/Windows/evil.txt",
    r"\Windows\evil.txt",
    "/etc/passwd",
    r"\\server\share\evil.txt",
    "D:evil.txt",
    "Drafts/a.txt:hidden",
    "CON",
    "Drafts/NUL.txt",
    "Drafts/com1.md",
    "",
]


@pytest.mark.parametrize("rel", ESCAPES)
def test_safe_path_rejects_escapes(lib, rel):
    with pytest.raises(LibraryError):
        safe_path(lib.root, rel)


def test_safe_path_rejects_sibling_with_same_prefix(tmp_path):
    root = tmp_path / "sandbox"
    root.mkdir()
    (tmp_path / "sandbox_evil").mkdir()
    with pytest.raises(LibraryError):
        safe_path(root, r"..\sandbox_evil\x.txt")


def test_safe_path_allows_inside(lib):
    p = safe_path(lib.root, "Drafts/notes.md")
    assert p == lib.root / "Drafts" / "notes.md"


def test_symlink_or_junction_out_is_rejected(lib, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    link = lib.root / "Drafts" / "escape"
    try:
        os.symlink(outside, link, target_is_directory=True)
    except OSError:
        # Symlinks need admin/developer mode on Windows; use a junction instead.
        import subprocess
        r = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(outside)], capture_output=True)
        if r.returncode != 0:
            pytest.skip("cannot create links here")
    with pytest.raises(LibraryError):
        safe_path(lib.root, "Drafts/escape/evil.txt")


# -- clean_title -------------------------------------------------------------

@pytest.mark.parametrize("title", [r"..\..\Windows\evil", "C:\\x", "a/b:c", "CON", "", "   ...   "])
def test_clean_title_is_always_a_plain_name(title):
    t = clean_title(title)
    assert t and not any(c in t for c in "\\/:.")
    assert t.upper() not in library.RESERVED_NAMES


# -- import ------------------------------------------------------------------

def test_import_copies_and_leaves_original(lib, sample):
    result = lib.import_files([str(sample)])
    assert result["imported"] == ["chapter one"]
    assert sample.exists()  # original untouched
    docs = lib.documents()
    assert [d.kind for d in docs] == ["source"]
    assert docs[0].id.startswith("src-")


def test_import_rejects_other_types(lib, tmp_path):
    exe = tmp_path / "evil.exe"
    exe.write_bytes(b"MZ")
    result = lib.import_files([str(exe)])
    assert result["imported"] == [] and result["skipped"]


def test_import_size_limit(lib, tmp_path, monkeypatch):
    monkeypatch.setattr(library, "MAX_IMPORT_BYTES", 10)
    big = tmp_path / "big.txt"
    big.write_text("x" * 100)
    assert lib.import_files([str(big)])["skipped"]


def test_import_name_clash_gets_suffix(lib, sample):
    lib.import_files([str(sample)])
    lib.import_files([str(sample)])
    assert sorted(d.title for d in lib.documents()) == ["chapter one", "chapter one-2"]


# -- IDs ---------------------------------------------------------------------

@pytest.mark.parametrize("bad", ["../Sources/x", r"C:\Windows\win.ini", "src-000000", "", None, "Sources/chapter one.txt"])
def test_ids_cannot_be_forged_into_paths(lib, sample, bad):
    lib.import_files([str(sample)])
    with pytest.raises(LibraryError):
        lib.get(bad)


# -- reading -----------------------------------------------------------------

def test_pagination(lib, tmp_path, monkeypatch):
    monkeypatch.setattr(library, "WORDS_PER_PAGE", 5)
    f = tmp_path / "long.txt"
    f.write_text("one two three four five six seven eight nine ten eleven", encoding="utf-8")
    lib.import_files([str(f)])
    doc = lib.documents()[0]
    _, page, total, text = lib.read_page(doc.id, 2)
    assert (page, total) == (2, 3)
    assert text == "six seven eight nine ten"
    _, page, _, _ = lib.read_page(doc.id, 99)  # clamps
    assert page == 3


def test_search(lib, sample):
    lib.import_files([str(sample)])
    hits = lib.search("fog")
    assert hits and "fog" in hits[0]["snippet"].lower()
    assert lib.search("zebra") == []


# -- writing -----------------------------------------------------------------

def test_create_draft_stays_in_drafts(lib):
    doc = lib.create_draft(r"..\..\Windows\System32\evil", "hello")
    assert doc.path.parent == lib.root / "Drafts"
    assert doc.path.read_text(encoding="utf-8") == "hello"


def test_create_draft_never_overwrites(lib):
    a = lib.create_draft("Notes", "first")
    b = lib.create_draft("Notes", "second")
    assert a.path != b.path
    assert a.path.read_text(encoding="utf-8") == "first"


def test_draft_size_limit(lib, monkeypatch):
    monkeypatch.setattr(library, "MAX_DRAFT_BYTES", 10)
    with pytest.raises(LibraryError):
        lib.create_draft("Big", "x" * 11)


def test_draft_count_limit(lib, monkeypatch):
    monkeypatch.setattr(library, "MAX_DRAFTS", 2)
    lib.create_draft("a", "x")
    lib.create_draft("b", "x")
    with pytest.raises(LibraryError):
        lib.create_draft("c", "x")


def test_empty_draft_rejected(lib):
    with pytest.raises(LibraryError):
        lib.create_draft("Empty", "   ")


def test_revise_draft_keeps_history(lib):
    doc = lib.create_draft("Story", "version one")
    lib.revise_draft(doc.id, "version two")
    assert doc.path.read_text(encoding="utf-8") == "version two"
    history = list((lib.root / ".history").iterdir())
    assert len(history) == 1 and history[0].read_text(encoding="utf-8") == "version one"


def test_sources_are_read_only(lib, sample):
    lib.import_files([str(sample)])
    src = lib.documents()[0]
    before = src.path.read_text(encoding="utf-8")
    with pytest.raises(LibraryError):
        lib.revise_draft(src.id, "overwritten!")
    assert src.path.read_text(encoding="utf-8") == before


def test_files_outside_allowed_types_are_invisible(lib):
    (lib.root / "Sources" / "script.py").write_text("print(1)")
    (lib.root / "Sources" / "notes.txt").write_text("hi")
    assert [d.title for d in lib.documents()] == ["notes"]


def test_onedrive_warning(monkeypatch, tmp_path):
    monkeypatch.setenv("OneDrive", str(tmp_path / "OneDrive"))
    assert library.onedrive_warning(tmp_path / "OneDrive" / "Docs")
    assert library.onedrive_warning(tmp_path / "Elsewhere") is None
