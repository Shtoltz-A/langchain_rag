from pathlib import Path

import typer
from ollama import Client

from .app import (
    PROJECT_ROOT,
    answer as answer_question,
    clear_index,
    index as index_documents,
)
from .console import Console


def _ensure_ollama_running() -> None:
    try:
        Client().list()
    except ConnectionError:
        raise typer.Exit(
            "Error: Ollama is not reachable. Start it with `ollama serve`."
        )


app = typer.Typer(help="Index documents and query them with RAG.")
c = Console()


@app.command()
def index(
    directory: Path = typer.Option(
        PROJECT_ROOT / "data/raw",
        "--directory",
        "-d",
        exists=True,
        file_okay=False,
        readable=True,
    ),
    clear: bool = typer.Option(
        False,
        "--clear",
        help="Delete ALL indexed data (vector store + manifest) and exit.",
    ),
) -> None:
    """Build and persist the document index (incremental per folder)."""
    _ensure_ollama_running()
    if clear:
        removed = clear_index()
        c.log(f"Index cleared: {removed} chunks removed, manifest file deleted.")
        return
    try:
        result = index_documents(directory)
    except ValueError as exc:
        c.error(str(exc))
        raise typer.Exit(code=1)
    c.log(
        f"Indexed {result.chunks_stored} new/changed chunks "
        f"({result.files_unchanged} files unchanged, "
        f"{result.files_deleted} removed from index)."
    )


@app.command()
def query(
    question: str = typer.Argument(..., help="Question to answer."),
) -> None:
    """Answer a question using the persisted index."""
    _ensure_ollama_running()
    c.success(answer_question(question))


if __name__ == "__main__":
    app()
