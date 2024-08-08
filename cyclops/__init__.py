from enum import Enum
from pathlib import Path


class ModelChoice(str, Enum):
    HOG = "hog"
    CNN = "cnn"


DEFAULT_ENCODINGS_PATH: Path = Path("output/encodings.pkl")
BOUNDING_BOX_COLOR: str = "blue"
TEXT_COLOR: str = "white"
