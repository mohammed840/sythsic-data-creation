class ModelError(RuntimeError):
    """A sanitized local error; never includes provider response bodies."""


class ModelResponseError(ModelError):
    """A model response failed parsing or shape validation."""
