"""Preflight checks shared by every workflow, independent of any front end.

Rule 6: no Typer and no process exits here. A failed check raises a domain
error and the caller decides how to report it — the CLI prints
``Console.error`` and exits 1, a future UI shows a banner.
"""

from ollama import Client, RequestError, ResponseError

from ..errors import OllamaUnavailableError


def ensure_ollama_running() -> None:
    """Confirm the Ollama daemon answers ``list``.

    Embeddings always run on Ollama, so indexing and answering both need a
    live daemon no matter which provider serves chat. Only connectivity is
    checked, never model names — a missing model is a different failure and
    surfaces later with the provider's own message.

    Raises:
        OllamaUnavailableError: the daemon refused the connection, timed out,
            or answered with an error; the client error is chained as
            ``__cause__``.
    """
    try:
        Client().list()
    except (ConnectionError, RequestError, ResponseError) as exc:
        # Covers more than a refused connection: RequestError/ResponseError
        # surface a daemon that is up but erroring. ``ConnectionError`` is the
        # *builtin* — ollama re-raises httpx's ConnectError as one and exports
        # no error type of that name, so do not "fix" this into the import
        # above: daemon-down detection would silently stop being caught.
        raise OllamaUnavailableError(
            "Ollama is not reachable. Start it with `ollama serve`."
        ) from exc
