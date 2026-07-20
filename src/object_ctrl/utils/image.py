"""
Utilities for working with images
"""
from io import BytesIO
from pathlib import Path

import numpy as np
from IPython.display import Image as IPyImage
from IPython.display import display as ipy_display
from PIL import Image as PILImage


def display(
    image: Path | str | IPyImage | PILImage.Image | np.ndarray,
    width: int | None = None,
    format: str | None = None,
) -> None:
    """
    Display an image in a Jupyter notebook.
    """
    if isinstance(image, (Path, str)):
        image = IPyImage(filename=str(image), width=width)
    if isinstance(image, np.ndarray):
        image = PILImage.fromarray(image)
    if isinstance(image, PILImage.Image):
        buffer = BytesIO()
        image.save(buffer, format=format or "PNG")
        image = IPyImage(data=buffer.getvalue(), format=format or "PNG", width=width)
    if isinstance(image, IPyImage) and width is not None:
        image.width = width

    ipy_display(image)
