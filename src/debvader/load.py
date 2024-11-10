from pathlib import Path

import numpy as np

__all__ = ["DATADIR", "AVAILABLE_FILES", "load_images", "load_fields"]

_BASEDIR = Path(__file__).parent.resolve()
DATADIR = _BASEDIR.joinpath("data")
_IMG_DIR = DATADIR.joinpath("dc2_imgs")
_FIELD_DIR = _IMG_DIR.joinpath("field")
AVAILABLE_FILES = [f.name for f in _FIELD_DIR.iterdir() if f.is_file()]


def load_images() -> np.ndarray:
    """Load the images from disk using memory mapping."""
    image_path = _IMG_DIR.joinpath("imgs_dc2.npy")
    return np.load(image_path, mmap_mode="r")


def load_fields(filename: str, allow_pickle: bool = False) -> np.ndarray:
    """Load the fields from disk using memory mapping."""
    # Normalise filename to remove any file extensions
    filename = filename.replace(".npy", "")
    filename = f"{filename}.npy"
    # Ensure file exists
    if filename not in AVAILABLE_FILES:
        raise ValueError(
            f"File {filename}.npy does not exist. The list of available files is {AVAILABLE_FILES}"
        )

    field_path = _FIELD_DIR.joinpath(filename)
    return np.load(field_path, mmap_mode="r", allow_pickle=allow_pickle)
