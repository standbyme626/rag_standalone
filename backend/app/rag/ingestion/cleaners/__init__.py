from .dedup import DedupCleaner  # noqa: F401
from .noise_filter import NoiseFilterCleaner  # noqa: F401
from .pii_redactor import PIIRedactorCleaner  # noqa: F401

_CLEANER_MAP: dict[str, type] = {
    "dedup": DedupCleaner,
    "noise_filter": NoiseFilterCleaner,
    "pii_redactor": PIIRedactorCleaner,
}


def get_cleaner(name: str):
    cls = _CLEANER_MAP.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown cleaner: {name}. Available: {list(_CLEANER_MAP.keys())}"
        )
    return cls()
