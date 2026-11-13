from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "samples"


@pytest.fixture
def sample_bytes():
    def load(name: str) -> bytes:
        return (SAMPLES / name).read_bytes()

    return load
