from pathlib import Path
from typing import Annotated

import typer

from .config import PROJECT_ROOT
from .console import get_console
from .errors import OllamaUnavailableError, RagError
from .services import answer as answer_question
from .services import clear_index, ensure_ollama_running
from .services import index as index_documents

app = typer.Typer(help="Index documents and query them with RAG.")
c = get_console()


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


def _require_ollama() -> None:
    """Exit this run with a readable message unless Ollama answers.

    The check itself lives in ``services/health.py`` and raises; translating
    that into terminal output plus an exit code is this layer's job (rule 6).
    """
    try:
        ensure_ollama_running()
    except OllamaUnavailableError as exc:
        c.error(str(exc))
        c.debug(f"Ollama client error: {exc.__cause__}")
        raise typer.Exit(code=1)


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
    _require_ollama()
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
    except (RagError, ValueError) as exc:
        # RagError = the app's own verdict on the run (a misconfigured .env for
        # the contextual summarizer); ValueError = validation that refused the
        # root or a chunking setting. Both are the user's to fix, so both end
        # the run with this one line instead of a traceback.
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
    _require_ollama()
    try:
        answer = answer_question(question)
    except (RagError, ValueError) as exc:
        # LLM/provider misconfiguration (unknown provider, missing gateway
        # settings) and an unknown retrieval mode end the run cleanly, the same
        # way `index` handles it.
        c.error(str(exc))
        raise typer.Exit(code=1)
    c.success(answer)


if __name__ == "__main__":
    app()
