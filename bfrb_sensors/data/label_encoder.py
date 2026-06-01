"""Deterministic, sorted gesture-label encoder with JSON persistence."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LabelEncoder:
    """Bidirectional mapping between gesture strings and integer ids.

    Indices are assigned in lexicographic order of the gesture strings so the
    mapping is reproducible from the data alone.
    """

    label_to_id: dict[str, int]
    id_to_label: dict[int, str]

    def encode(self, label: str) -> int:
        try:
            return self.label_to_id[label]
        except KeyError as exc:
            raise ValueError(f"unknown gesture {label!r}") from exc

    def decode(self, idx: int) -> str:
        try:
            return self.id_to_label[idx]
        except KeyError as exc:
            raise ValueError(f"unknown gesture id {idx!r}") from exc

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.label_to_id, indent=2, sort_keys=True) + "\n")
        logger.info("Saved label encoder with %d classes to %s", len(self.label_to_id), path)

    @classmethod
    def load(cls, path: Path) -> LabelEncoder:
        payload: dict[str, int] = json.loads(Path(path).read_text())
        return cls(
            label_to_id=payload,
            id_to_label={idx: label for label, idx in payload.items()},
        )

    @property
    def n_classes(self) -> int:
        return len(self.label_to_id)


def build_label_encoder(labels: Iterable[str]) -> LabelEncoder:
    unique = sorted(set(labels))
    label_to_id = {label: idx for idx, label in enumerate(unique)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    logger.info("Built label encoder with %d unique classes", len(unique))
    return LabelEncoder(label_to_id=label_to_id, id_to_label=id_to_label)
