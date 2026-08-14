"""Safe local image decoding for Dexter's single read_image(path) tool."""

from __future__ import annotations

import io
import warnings
from dataclasses import dataclass

from PIL import Image, ImageOps, UnidentifiedImageError

from filesystem import resolve_allowed_path


@dataclass(frozen=True)
class ModelImage:
    data: bytes
    width: int
    height: int


def _png_compatible(image: Image.Image) -> Image.Image:
    # Preserve alpha where present; normalize modes that PNG/Pillow cannot save
    # reliably as a static model image.
    if image.mode in {"1", "L", "LA", "P", "RGB", "RGBA"}:
        return image
    if "A" in image.getbands():
        return image.convert("RGBA")
    return image.convert("RGB")


def load_image_for_model(path: str) -> ModelImage:
    """Decode an allowed image by content, orient it, and return static PNG bytes."""
    image_path = resolve_allowed_path(path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image does not exist: {image_path}")
    if not image_path.is_file():
        raise ValueError(f"Image path is not a regular file: {image_path}")

    try:
        # Pillow's decompression-bomb warning is promoted to an exception here;
        # DecompressionBombError already raises normally.
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(image_path) as opened:
                # Animated inputs intentionally use the first frame only.
                if getattr(opened, "is_animated", False):
                    opened.seek(0)
                opened.load()  # force decoding before any ImageContent exists
                frame = opened.copy()

        frame = ImageOps.exif_transpose(frame)
        frame.load()
        frame = _png_compatible(frame)

        output = io.BytesIO()
        frame.save(output, format="PNG")
        data = output.getvalue()
        if not data:
            raise ValueError("Decoded image produced empty PNG data.")
        return ModelImage(data=data, width=frame.width, height=frame.height)
    except Image.DecompressionBombWarning as exc:
        raise ValueError(f"Image is too large to decode safely: {image_path}") from exc
    except Image.DecompressionBombError as exc:
        raise ValueError(f"Image is too large to decode safely: {image_path}") from exc
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError(f"Could not decode image safely: {image_path}: {exc}") from exc
