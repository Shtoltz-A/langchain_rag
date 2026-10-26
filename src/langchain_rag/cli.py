from pathlib import Path

import typer
from ollama import Client

from .app import PROJECT_ROOT, answer as answer_question, index as index_documents
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
) -> None:
    """Build and persist the document index."""
    _ensure_ollama_running()
    count = index_documents(directory)
    c.log(f"Indexed {count} chunks.")


@app.command()
def query(
    question: str = typer.Argument(..., help="Question to answer."),
) -> None:
    """Answer a question using the persisted index."""
    _ensure_ollama_running()
    c.success(answer_question(question))


if __name__ == "__main__":
    app()