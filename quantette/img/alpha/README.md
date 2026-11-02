This dataset contains images with an alpha channel, for benchmarking and testing quantization of images with transparency. The images were chosen to cover a variety of common uses of transparency:

- Photo cutouts, illustrations, vector art, and logos, which are mostly fully opaque or fully transparent with anti-aliased edges.
- 3D renders and emoji, which have soft edges and, in the case of `dna.png`, large partially transparent areas.
- Synthetic images made by adding smooth alpha masks to images from the [CQ100](../CQ100/) dataset, since smooth fades (e.g., shadows, glows, and vignettes) are common but rare in other freely licensed images. See `generate_synthetic.py` for how these were made.

The images were gathered from [Wikimedia Commons](https://commons.wikimedia.org/), [Microsoft's Fluent Emoji](https://github.com/microsoft/fluentui-emoji), and the CQ100 dataset. Other than the synthetic images, the only modification was renaming the files.

Each image is available under the license of its source: either CC0, public domain, the MIT license, or (for `king_penguin_cutout.png`) CC BY 2.0. For each image's license, author, and source URL, as well as the proportion of fully transparent, fully opaque, and partially transparent pixels, see the `metadata.csv` file.
