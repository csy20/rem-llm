"""Tests for codebase indexer."""

import tempfile
from pathlib import Path

from remllm.context.indexer import CodebaseIndexer, CodeChunk


def test_indexer_chunk_typescript():
    indexer = CodebaseIndexer()
    code = """
export function hello() {
  return 'world'
}

export class MyClass {
  greet() { return 'hi'; }
}
"""
    chunks = indexer._chunk_typescript(Path("test.ts"), code)
    assert len(chunks) >= 2
    names = [c.name for c in chunks]
    assert "hello" in names
    assert "MyClass" in names


def test_indexer_chunk_python():
    indexer = CodebaseIndexer()
    code = """
def hello():
    return 'world'

class MyClass:
    def greet(self):
        return 'hi'
"""
    chunks = indexer._chunk_python(Path("test.py"), code)
    assert len(chunks) >= 2
    names = [c.name for c in chunks]
    assert "hello" in names
    assert "MyClass" in names


def test_indexer_chunk_prisma():
    indexer = CodebaseIndexer()
    code = """
model User {
  id Int @id
  name String
}

model Post {
  id Int @id
  title String
}
"""
    chunks = indexer._chunk_prisma(Path("schema.prisma"), code)
    assert len(chunks) == 2
    names = [c.name for c in chunks]
    assert "User" in names
    assert "Post" in names


def test_indexer_embed_text():
    indexer = CodebaseIndexer()
    embedding = indexer._embed_text("hello world")
    assert len(embedding) == 384
    assert all(isinstance(v, float) for v in embedding)


def test_indexer_cosine_sim():
    indexer = CodebaseIndexer()
    a = [1.0, 0.0, 0.0]
    b = [1.0, 0.0, 0.0]
    assert abs(indexer._cosine_sim(a, b) - 1.0) < 0.01

    c = [0.0, 1.0, 0.0]
    assert abs(indexer._cosine_sim(a, c)) < 0.01


def test_index_directory(tmp_path: Path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "hello.ts").write_text(
        "export function hello() { return 'world'; }"
    )
    (tmp_path / "src" / "utils.py").write_text("def foo():\n    return 42")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "dep.ts").write_text("export const x = 1")

    indexer = CodebaseIndexer(tmp_path / "index.json")
    count = indexer.index(tmp_path)
    assert count >= 2
    assert count < 3  # node_modules skipped

    indexer2 = CodebaseIndexer(tmp_path / "index.json")
    indexer2._load()
    assert len(indexer2.chunks) == count


def test_retrieve():
    indexer = CodebaseIndexer()
    indexer.chunks = [
        CodeChunk(
            path="a.ts",
            name="hello",
            chunk_type="function",
            content="export function hello() { return 'world' }",
            start_line=1,
            end_line=3,
            embedding=indexer._embed_text("hello world function"),
        ),
        CodeChunk(
            path="b.ts",
            name="goodbye",
            chunk_type="function",
            content="export function goodbye() { return 'bye' }",
            start_line=1,
            end_line=3,
            embedding=indexer._embed_text("goodbye function"),
        ),
    ]
    results = indexer.retrieve("hello world")
    assert len(results) > 0
    assert results[0].name == "hello"


def test_build_context_prompt():
    indexer = CodebaseIndexer()
    indexer.chunks = [
        CodeChunk(
            path="src/hello.ts",
            name="hello",
            chunk_type="function",
            content="export function hello() { return 'world' }",
            start_line=1,
            end_line=3,
            embedding=indexer._embed_text("hello function"),
        ),
    ]
    prompt = indexer.build_context_prompt("hello")
    assert "hello" in prompt
    assert "src/hello.ts" in prompt
    assert "```" in prompt
