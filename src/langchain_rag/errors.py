"""The application's own failure vocabulary, importable from every layer.

A leaf module on purpose (like ``config.py`` and ``console.py``): it imports
nothing from the package, so ``generation/``, ``services/``, ``retrieval/``
and the CLI can all raise or catch these types without a cycle — putting them
under ``services/`` would close a ``generation`` ⇄ ``services`` one, since the
workflows already import ``generation``.

``ConfigurationError`` deliberately does **not** subclass ``ValueError``: the
same code raises plain ``ValueError`` for internal validation (overlapping
index roots, retrieval-mode guards, rerank score parsing), and a front end
must be able to tell "this ``.env`` is wrong" from "an invariant failed"
without sniffing messages. Rationale and revisit conditions:
``.agents/rag-decisions.md``.
"""


class RagError(Exception):
    """Base for every failure the application raises deliberately.

    A front end catches this to handle all operational errors uniformly;
    library and OS failures keep their own types and are not wrapped.
    """


class ConfigurationError(RagError):
    """A required setting is missing or invalid for the chosen provider.

    Raised when a pipeline path is configured for a provider it cannot build
    a client for: unknown provider, unset gateway model, base URL or API key.
    The message names the ``.env`` key to fix.
    """


class OllamaUnavailableError(RagError):
    """The Ollama daemon is down, unreachable, or answering with an error.

    Raised by ``services.health.ensure_ollama_running()``; the client error
    that proved it is chained as ``__cause__`` for verbose diagnostics.
    """
