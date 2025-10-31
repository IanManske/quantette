# Examples

Below are some examples of `quantette` in action. The dissimilarity between the each image and the original is reported in the tables below using [`dssim`](https://crates.io/crates/dssim) (lower numbers are better). Each table starts with output from GIMP as a comparison.

Each output image was created like so:
- The GIMP output was creating using `Image > Mode > Indexed` with GIMP version `2.10.36`. For dithered outputs, `Floyd-Steinberg (reduced color bleeding)` was used.
- The `Wu (sRGB)` output was creating using `quantette`'s fastest quantization method. The default number of bins was used. This method does not easily support dithering and is omitted from the dithering section below.
- The `Wu (Oklab)` output is similar to the above but performed in the Oklab color space. It serves as an example mix between speed and accuracy. It also supports dithering. The default number of bins was used.
- The `k-means` output was creating using `quantette`'s most accurate quantization method running in the Oklab color space. The default `KmeansOptions` were used.

# Original Image

![Calaveras](../img/CQ100/img/calaveras.png)

# Without Dithering

## 16 Colors

| Method     | DSSIM    | Result                   |
| ---------- | -------- | ------------------------ |
| Gimp       | 0.063687 | ![](img/gimp_16.png)     |
| Wu (sRGB)  | 0.042151 | ![](img/wu_srgb_16.png)  |
| Wu (Oklab) | 0.038925 | ![](img/wu_oklab_16.png) |
| k-means    | 0.02628  | ![](img/kmeans_16.png)   |

## 64 Colors

| Method     | DSSIM    | Result                   |
| ---------- | -------- | ------------------------ |
| Gimp       | 0.017303 | ![](img/gimp_64.png)     |
| Wu (sRGB)  | 0.013167 | ![](img/wu_srgb_64.png)  |
| Wu (Oklab) | 0.007939 | ![](img/wu_oklab_64.png) |
| k-means    | 0.006328 | ![](img/kmeans_64.png)   |

## 256 Colors

| Method     | DSSIM    | Result                    |
| ---------- | -------- | ------------------------- |
| Gimp       | 0.004888 | ![](img/gimp_256.png)     |
| Wu (sRGB)  | 0.003432 | ![](img/wu_srgb_256.png)  |
| Wu (Oklab) | 0.002093 | ![](img/wu_oklab_256.png) |
| k-means    | 0.00155  | ![](img/kmeans_256.png)   |

# With Dithering

## 16 Colors

| Method     | DSSIM    | Result                          |
| ---------- | -------- | ------------------------------- |
| Gimp       | 0.050564 | ![](img/gimp_16_dither.png)     |
| Wu (Oklab) | 0.027656 | ![](img/wu_oklab_16_dither.png) |
| k-means    | 0.018208 | ![](img/kmeans_16_dither.png)   |

## 64 Colors

| Method     | DSSIM   | Result                          |
| ---------- | ------- | ------------------------------- |
| Gimp       | 0.01366 | ![](img/gimp_64_dither.png)     |
| Wu (Oklab) | 0.0052  | ![](img/wu_oklab_64_dither.png) |
| k-means    | 0.00424 | ![](img/kmeans_64_dither.png)   |

## 256 Colors

| Method     | DSSIM    | Result                           |
| ---------- | -------- | -------------------------------- |
| Gimp       | 0.004073 | ![](img/gimp_256_dither.png)     |
| Wu (Oklab) | 0.001417 | ![](img/wu_oklab_256_dither.png) |
| k-means    | 0.001061 | ![](img/kmeans_256_dither.png)   |
