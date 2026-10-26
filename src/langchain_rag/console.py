"""console.py — Consistent, colored terminal output using Typer."""

from __future__ import annotations

import textwrap
import time

import typer


class Console:
    """Print timestamped messages, headings, and multiline blocks.

    Use this class to keep terminal output consistent across an application.
    Verbose mode enables debug messages and disables block truncation.
    """

    LEVEL_COLORS = {
        "INFO": typer.colors.CYAN,
        "WARNING": typer.colors.YELLOW,
        "SUCCESS": typer.colors.GREEN,
        "ERROR": typer.colors.RED,
        "DEBUG": typer.colors.BRIGHT_BLACK,
    }

    def __init__(
        self,
        verbose: bool = False,
        color: bool | None = None,
    ) -> None:
        """Configure output behavior and start the elapsed-time clock.

        Args:
            verbose: Show debug output and complete multiline blocks.
            color: Force colors on (True), off (False), or let Typer
                detect terminal support (None).

        Use cases:
            - Enable verbose output with a CLI flag.
            - Disable colors for plain-text output.
            - Force colors in an environment that supports ANSI sequences.
        """
        self.verbose = verbose
        self.color = color
        self._start = time.perf_counter()

    def _elapsed(self) -> str:
        """Return the time elapsed since this console was created.

        Use cases:
            - Timestamp messages relative to application startup.
            - Spot long gaps between operations.

        This is an internal formatting helper, not a precise profiler.
        """
        return f"{time.perf_counter() - self._start:6.1f}s"

    def _echo(self, message: str, *, err: bool = False) -> None:
        """Write a message using the configured color behavior.

        Args:
            message: Text to print, optionally containing styled segments.
            err: Write to stderr instead of stdout.

        Use cases:
            - Centralize terminal writes.
            - Keep color handling consistent across all methods.
        """
        typer.echo(message, err=err, color=self.color)

    def log(
        self,
        message: str,
        level: str = "INFO",
        icon: str = "•",
    ) -> None:
        """Print a timestamped message with a level and icon.

        Args:
            message: Text to display.
            level: Label used to select the color, case-insensitively.
                Unknown levels use white.
            icon: Symbol displayed before the message.

        Use cases:
            - Print a custom event such as log("Reading file", icon="→").
            - Provide the shared formatting for convenience methods.

        WARNING and ERROR messages are written to stderr.
        Direct calls with level="DEBUG" are not filtered by verbose mode;
        use debug() for conditional debug output.
        """
        level = level.upper()
        color = self.LEVEL_COLORS.get(level, typer.colors.WHITE)

        timestamp = typer.style(self._elapsed(), dim=True)
        label = typer.style(f"[{level:<7}]", fg=color, bold=True)
        body = typer.style(f"{icon} {message}", fg=color)

        self._echo(
            f"{timestamp} {label} {body}",
            err=level in {"WARNING", "ERROR"},
        )

    def info(self, message: str) -> None:
        """Print a normal informational message to stdout.

        Use cases:
            - Announce application startup.
            - Report the file or directory currently being processed.
            - Describe progress without implying success or failure.
        """
        self.log(message)

    def warning(self, message: str) -> None:
        """Print a warning to stderr without interrupting execution.

        Use cases:
            - Report a missing optional configuration file.
            - Explain that a default value or fallback will be used.
            - Highlight deprecated settings.
        """
        self.log(message, level="WARNING", icon="!")

    def success(self, message: str) -> None:
        """Print a successful outcome to stdout.

        Use cases:
            - Confirm a file was saved.
            - Report that validation passed.
            - Announce that an operation completed.
        """
        self.log(message, level="SUCCESS", icon="✓")

    def error(self, message: str) -> None:
        """Print an error to stderr without raising or exiting.

        Use cases:
            - Report an unreadable file.
            - Display a failed operation.
            - Present a concise explanation of a caught exception.

        The caller remains responsible for recovery or process exit.
        """
        self.log(message, level="ERROR", icon="✗")

    def debug(self, message: str) -> None:
        """Print a diagnostic message only when verbose mode is enabled.

        Use cases:
            - Show resolved paths or configuration decisions.
            - Report intermediate values.
            - Explain why a particular execution branch was selected.

        Avoid including credentials or other sensitive data.
        """
        if self.verbose:
            self.log(message, level="DEBUG")

    def banner(self, title: str) -> None:
        """Print a prominent heading surrounded by horizontal rules.

        Use cases:
            - Display the application name at startup.
            - Mark the beginning of a major workflow.
            - Separate independent runs in terminal output.
        """
        line = "═" * 78
        self._echo(
            typer.style(
                f"\n{line}\n  {title}\n{line}",
                fg=typer.colors.MAGENTA,
                bold=True,
            )
        )

    def section(self, title: str) -> None:
        """Print a smaller heading for a group of related operations.

        Use cases:
            - Separate configuration, processing, and summary stages.
            - Group output by input file or task.
            - Improve readability without adding a full banner.
        """
        self._echo(
            typer.style(
                f"\n── {title} " + "─" * max(0, 74 - len(title)),
                fg=typer.colors.MAGENTA,
                bold=True,
            )
        )

    def block(
        self,
        title: str,
        content: str,
        level: str = "INFO",
        max_lines: int | None = 40,
    ) -> None:
        """Print multiline content inside a decorative border.

        Args:
            title: Label displayed on the top border.
            content: Text to display, preserving line indentation.
            level: Selects the border color and output stream.
            max_lines: Maximum content lines in non-verbose mode.
                None or a non-positive value disables truncation.

        Use cases:
            - Display configuration, JSON, or code snippets.
            - Show command output or validation errors.
            - Preview a large document without flooding the terminal.

        Verbose mode always shows all lines. Long lines are not wrapped.
        WARNING and ERROR blocks are written to stderr.
        """
        level = level.upper()
        color = self.LEVEL_COLORS.get(level, typer.colors.WHITE)
        err = level in {"WARNING", "ERROR"}
        lines = content.rstrip().splitlines() or ["<empty>"]

        if (
                not self.verbose
                and max_lines is not None
                and 0 < max_lines < len(lines)
        ):
            hidden = len(lines) - max_lines
            lines = lines[:max_lines] + [
                f"... ({hidden} more lines; enable verbose mode to see all)"
            ]

        top = f"┌─ {title} " + "─" * max(0, 72 - len(title))
        self._echo(typer.style(top, fg=color), err=err)

        border = typer.style("│", fg=color)
        for line in lines:
            self._echo(f"{border} {line}", err=err)

        self._echo(typer.style("└" + "─" * 76, fg=color), err=err)

    def debug_block(self, title: str, content: str) -> None:
        """Print a complete diagnostic block only in verbose mode.

        Common leading indentation is removed for readability.

        Use cases:
            - Display detailed configuration dumps.
            - Inspect full command responses.
            - Print diagnostic text written as an indented multiline string.

        Avoid including credentials or other sensitive data.
        """
        if self.verbose:
            self.block(
                title,
                textwrap.dedent(content),
                level="DEBUG",
                max_lines=None,
            )