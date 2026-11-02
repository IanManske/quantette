"""Generates the synthetic images in this dataset by adding alpha masks to CQ100 images.

Usage (from this directory, requires Pillow): python generate_synthetic.py
"""

from PIL import Image

CQ100 = "../CQ100/img"


def vignette(x, y, w, h):
    """A soft radial fade, fully opaque in the center and fully transparent at the corners."""
    d = (((x - w / 2) / w) ** 2 + ((y - h / 2) / h) ** 2) ** 0.5
    return (0.6 - d) / 0.3


def fade(x, y, w, h):
    """A linear fade from fully transparent on the left to fully opaque on the right."""
    return x / (w - 1)


def feathered_ellipse(x, y, w, h):
    """An elliptical cutout with a soft, feathered edge."""
    r = (((x - w / 2) / (0.42 * w)) ** 2 + ((y - h / 2) / (0.45 * h)) ** 2) ** 0.5
    return (1 - r) / 0.08


IMAGES = [
    ("fruits", "fruits_vignette", vignette),
    ("cosmic_vista", "cosmic_vista_fade", fade),
    ("king_penguin", "king_penguin_cutout", feathered_ellipse),
]

for source, name, mask in IMAGES:
    image = Image.open(f"{CQ100}/{source}.png").convert("RGBA")
    w, h = image.size
    alpha = Image.new("L", (w, h))
    alpha.putdata(
        [round(255 * min(max(mask(x, y, w, h), 0.0), 1.0)) for y in range(h) for x in range(w)]
    )
    image.putalpha(alpha)
    image.save(f"img/{name}.png", optimize=True)
