"""Logging, warnings, and progress-bar control for AFRLib."""
from typing import Optional, TextIO
import contextlib
import logging
import os
import sys
import threading
import warnings

# Package logger and verbosity

_LOGGER_NAME = "afrlib"
_lock = threading.Lock()
_default_handler: Optional[logging.Handler] = None

_ENV_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}


def _get_default_level() -> int:
    env = os.environ.get("AFRLIB_VERBOSITY", "").strip().lower()
    if env:
        if env in _ENV_LEVELS:
            return _ENV_LEVELS[env]
        warnings.warn(
            f"Unknown AFRLIB_VERBOSITY={env!r}; expected one of {sorted(_ENV_LEVELS)}. "
            "Falling back to 'info'."
        )
    return logging.INFO


def _configure_root() -> logging.Logger:
    """Attach the single stderr handler to the ``afrlib`` logger (idempotent)."""
    global _default_handler
    with _lock:
        root = logging.getLogger(_LOGGER_NAME)
        if _default_handler is None:
            _default_handler = logging.StreamHandler(sys.stderr)
            _default_handler.setFormatter(
                logging.Formatter("[%(levelname)s|%(name)s] %(message)s")
            )
            root.addHandler(_default_handler)
            root.setLevel(_get_default_level())
            root.propagate = False
        return root


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a logger under the ``afrlib`` hierarchy; call it as ``get_logger(__name__)`` so
    per-module filtering works, or with ``None`` for the package logger.
    """
    _configure_root()
    if name is None or name == _LOGGER_NAME:
        return logging.getLogger(_LOGGER_NAME)
    if not name.startswith(_LOGGER_NAME + "."):
        name = f"{_LOGGER_NAME}.{name}"
    return logging.getLogger(name)


def get_verbosity() -> int:
    """Return the current verbosity level of the ``afrlib`` root logger."""
    return _configure_root().getEffectiveLevel()


def set_verbosity(level: int) -> None:
    """Set the global AFRLib verbosity (a ``logging`` level such as ``logging.INFO``)."""
    _configure_root().setLevel(level)


def set_verbosity_debug() -> None:
    set_verbosity(logging.DEBUG)


def set_verbosity_info() -> None:
    set_verbosity(logging.INFO)


def set_verbosity_warning() -> None:
    set_verbosity(logging.WARNING)


def set_verbosity_error() -> None:
    set_verbosity(logging.ERROR)


def add_file_handler(path: str, level: int = logging.DEBUG) -> logging.Handler:
    """Also write AFRLib logs to ``path`` and return the handler, which :func:`remove_handler`
    detaches; ``level`` filters this handler only.
    """
    handler = logging.FileHandler(os.path.expanduser(path))
    handler.setLevel(level)
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s|%(name)s] %(message)s")
    )
    _configure_root().addHandler(handler)
    return handler


def remove_handler(handler: logging.Handler) -> None:
    """Detach a handler returned by :func:`add_file_handler`."""
    _configure_root().removeHandler(handler)


# Warnings

class AFRLibWarning(UserWarning):
    """Category for AFRLib runtime warnings (filterable via ``warnings``)."""


def warn(msg: str, stacklevel: int = 2) -> None:
    """Emit an :class:`AFRLibWarning` for a recoverable anomaly."""
    warnings.warn(msg, AFRLibWarning, stacklevel=stacklevel + 1)


# Repeated warnings (e.g. one per image in a batch) should all be shown.
warnings.filterwarnings("always", category=AFRLibWarning)


# Progress bars

_progress_bars_enabled = os.environ.get(
    "AFRLIB_DISABLE_PROGRESS_BARS", ""
).strip().lower() not in ("1", "true", "yes", "on")


def enable_progress_bars() -> None:
    """Globally enable AFRLib progress bars."""
    global _progress_bars_enabled
    _progress_bars_enabled = True


def disable_progress_bars() -> None:
    """Globally disable AFRLib progress bars."""
    global _progress_bars_enabled
    _progress_bars_enabled = False


def are_progress_bars_enabled() -> bool:
    return _progress_bars_enabled


class _NullBar:
    """No-op stand-in for a ``tqdm`` bar, supporting iteration, the context-manager protocol and
    the common bar methods.
    """

    def __init__(self, iterable=None, total=None, **kwargs):
        self._iterable = iterable
        self.total = total
        self.n = 0

    def __iter__(self):
        if self._iterable is None:
            return iter(())
        return iter(self._iterable)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass

    def update(self, n: int = 1) -> None:
        self.n += n

    def close(self) -> None:
        pass

    def refresh(self) -> None:
        pass

    def set_description(self, *args, **kwargs) -> None:
        pass

    def set_postfix(self, *args, **kwargs) -> None:
        pass

    def write(self, *args, **kwargs) -> None:
        pass


def tqdm(iterable=None, *args, **kwargs):
    """Drop-in ``tqdm`` replacement that honors the global progress-bar switch and returns a
    no-op bar when bars are off or ``tqdm`` is missing.
    """
    if not _progress_bars_enabled:
        return _NullBar(iterable, total=kwargs.get("total"))
    try:
        from tqdm.auto import tqdm as _tqdm
    except ImportError:
        return _NullBar(iterable, total=kwargs.get("total"))

    return _tqdm(iterable, *args, **kwargs)


# Output suppression for vendored code

class _LoggerWriter:
    """File-like object that forwards complete lines to a logger."""

    def __init__(self, logger: logging.Logger, level: int):
        self._logger = logger
        self._level = level
        self._buf = ""

    def write(self, msg: str) -> int:
        self._buf += msg
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if line.strip():
                self._logger.log(self._level, line)
        return len(msg)

    def flush(self) -> None:
        if self._buf.strip():
            self._logger.log(self._level, self._buf)
        self._buf = ""


@contextlib.contextmanager
def suppress_output(logger: Optional[logging.Logger] = None, level: int = logging.DEBUG):
    """Redirect stdout and stderr inside the block to ``logger`` at ``level``, so noisy vendored
    code can stay unmodified.
    """
    logger = logger or get_logger()
    out: TextIO = _LoggerWriter(logger, level)  # type: ignore[assignment]
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
        try:
            yield
        finally:
            out.flush()
