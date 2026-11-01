from pathlib import Path
from typing import Annotated

import typer
from ollama import Client, RequestError, ResponseError

from .app import (
    answer as answer_question,
)
from .app import (
    clear_index,
)
from .app import (
    index as index_documents,
)
from .config import PROJECT_ROOT
from .console import get_console

app = typer.Typer(help="Index documents and query them with RAG.")
c = get_console()


def _ensure_ollama_running() -> None:
    """Abort the run unless the Ollama daemon answers ``list``.

    Covers more than a refused connection: ``ResponseError``/``RequestError``
    surface a daemon that is up but erroring, so both end the run with a
    Console error instead of a raw traceback.
    """
    try:
        Client().list()
    except (ConnectionError, RequestError, ResponseError) as exc:
        c.error("Ollama is not reachable. Start it with `ollama serve`.")
        c.debug(f"Ollama client error: {exc}")
        raise typer.Exit(code=1)


@app.callback()
def main(
    verbose: Annotated[
        bool | None,
        typer.Option(
            "--verbose",
            "-v",
            help="Show debug output (retrieval lines, context dumps); "
            "overrides the verbose setting for this run.",
        ),
    ] = None,
) -> None:
    """Apply the verbosity override before any subcommand runs.

    Flag absent → None → the shared console keeps the setting's value;
    ``--verbose`` forces debug output on for the whole run.
    """
    if verbose is not None:
        get_console().verbose = verbose


@app.command()
def index(
    directory: Path = typer.Option(
        PROJECT_ROOT / "data/raw",
        "--directory",
        "-d",
        file_okay=False,
        readable=True,
    ),
    clear: bool = typer.Option(
        False,
        "--clear",
        help="Delete ALL indexed data (vector store + manifest) and exit. "
        "The chunk-summary cache (data/chunk_summaries.json) is kept.",
    ),
) -> None:
    """Build and persist the document index (incremental per folder)."""
    # Existence is validated here, not by Click: `--clear` never reads the
    # directory, so clearing must work even after data/raw is gone.
    if not clear and not directory.exists():
        c.error(f"Directory '{directory}' does not exist.")
        raise typer.Exit(code=1)
    _ensure_ollama_running()
    if clear:
        result = clear_index()
        pruned = (
            f", {result.folders_pruned} stale segment folder(s) pruned"
            if result.folders_pruned
            else ""
        )
        c.log(
            f"Index cleared: {result.chunks_removed} chunks removed, "
            f"manifest file deleted{pruned}."
        )
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
    try:
        answer = answer_question(question)
    except ValueError as exc:
        # LLM/provider misconfiguration (unknown provider, missing gateway
        # settings) ends the run cleanly, the same way `index` handles it.
        c.error(str(exc))
        raise typer.Exit(code=1)
    c.success(answer)


if __name__ == "__main__":
    app()
