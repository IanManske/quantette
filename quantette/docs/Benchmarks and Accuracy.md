# Benchmarks and Accuracy

Below are some comparisons between `quantette` and some other libraries. Each library has different situations and/or options that can make them perform better or worse, but these limited comparisons should give a rough point of reference. The sections below cover opaque images using `Pipeline`. See [Images with Transparency](#images-with-transparency) for images with an alpha channel using `RgbaPipeline`.

# Setup

The settings and libraries being compared are:

- `Wu (sRGB)`: `quantette`'s Wu's color quantizer implementation running in the sRGB color space. The default number of bins was used.
- `Wu (Oklab)`: `quantette`'s Wu's color quantizer implementation running in the Oklab color space. The default number of bins was used.
- `k-means`: `quantette`'s k-means color quantizer running in the Oklab color space. The default `KmeansOptions` were used.
- `imagequant` version `4.4.1` run with the default library options (quality of `100`).
- `color_quant` version `1.1.0` run with a `sample_frac` of `10`.
- `exoquant` version `0.2.0` run without k-means optimization, since it would otherwise take way too long.

The "Time" tables below provide the total time, in milliseconds, for quantization and remapping as reported by the `cli` binary in [`examples/`](../examples/) with the `--verbose` flag. 30 trials were run and averaged for each data point. The `Wu`, `k-means`, and `imagequant` columns used 4 threads, while `color_quant` and `exoquant` only support single-threaded execution. So, multiply or divide by 4 as you see fit.

The "DSSIM" tables list the DSSIM values as reported by the `accuracy` binary found in [`examples/`](../examples/) using the [`dssim`](https://crates.io/crates/dssim) crate. Note that `exoquant` results are not deterministic, since it uses `rand::random()`.

`color_quant` does not have dithering, so it is not included in the `With Dithering` tables below. Also, `Wu (sRGB)` does not easily support dithering and is omitted from the dithering tables below.

All results below are for 256 colors, and the `-C target-feature=+avx2` `rustc` flags were used. The images used can be found in [img/unsplash/](../img/unsplash/).

# Without Dithering

## Time

| Image                | Dimensions | Wu (sRGB) | Wu (Oklab) | k-means | imagequant | color_quant | exoquant |
|:-------------------- |:---------- | ---------:| ----------:| -------:| ----------:| -----------:| --------:|
| Akihabara            | 5663x3769  | 25        | 80         | 97      | 1044       | 2155        | 4815     |
| Boothbay             | 6720x4480  | 34        | 83         | 91      | 694        | 2801        | 4836     |
| Bryggen              | 5508x3098  | 21        | 46         | 53      | 530        | 1122        | 2155     |
| Futuristic           | 6000x4000  | 30        | 72         | 76      | 253        | 1572        | 2598     |
| Hokkaido             | 6000x4000  | 31        | 66         | 74      | 513        | 2064        | 3652     |
| Jewel Changi         | 6000x4000  | 28        | 69         | 75      | 442        | 1733        | 3156     |
| Lake Atitlan         | 5112x3408  | 21        | 64         | 78      | 959        | 1686        | 3840     |
| Louvre               | 6056x4000  | 29        | 68         | 76      | 474        | 2201        | 3438     |
| Rosetta Nebula       | 5828x3885  | 28        | 60         | 67      | 319        | 1900        | 2839     |
| Termas Geometricas   | 5472x3648  | 23        | 57         | 66      | 825        | 2002        | 3456     |
| Yosemite Tunnel View | 5580x3720  | 24        | 57         | 64      | 899        | 2104        | 3759     |
|                      |            |           |            |         |            |             |          |
| Akihabara            | 1920x1278  | 3         | 10         | 31      | 266        | 262         | 953      |
| Boothbay             | 1920x1280  | 3         | 10         | 32      | 175        | 224         | 670      |
| Bryggen              | 1920x1080  | 3         | 9          | 28      | 96         | 147         | 359      |
| Futuristic           | 1920x1280  | 3         | 10         | 32      | 93         | 156         | 327      |
| Hokkaido             | 1920x1280  | 3         | 10         | 32      | 156        | 225         | 581      |
| Jewel Changi         | 1920x1280  | 3         | 10         | 32      | 124        | 193         | 446      |
| Lake Atitlan         | 1920x1280  | 3         | 10         | 32      | 247        | 235         | 869      |
| Louvre               | 1920x1268  | 3         | 10         | 31      | 143        | 216         | 533      |
| Rosetta Nebula       | 1920x1280  | 3         | 10         | 31      | 105        | 223         | 387      |
| Termas Geometricas   | 1920x1280  | 3         | 10         | 31      | 165        | 261         | 623      |
| Yosemite Tunnel View | 1920x1280  | 3         | 10         | 31      | 159        | 238         | 589      |

## DSSIM

| Image                | Dimensions | Wu (sRGB) | Wu (Oklab) | k-means  | imagequant | color_quant | exoquant |
|:-------------------- |:---------- |:--------- |:---------- |:-------- |:---------- |:----------- |:-------- |
| Akihabara            | 5663x3769  | 0.007178  | 0.005379   | 0.003991 | 0.004436   | 0.007313    | 0.005661 |
| Boothbay             | 6720x4480  | 0.004569  | 0.004305   | 0.002464 | 0.002419   | 0.005882    | 0.003464 |
| Bryggen              | 5508x3098  | 0.003877  | 0.00618    | 0.002001 | 0.00207    | 0.003315    | 0.002234 |
| Futuristic           | 6000x4000  | 0.001906  | 0.002126   | 0.000658 | 0.000608   | 0.001026    | 0.000663 |
| Hokkaido             | 6000x4000  | 0.003467  | 0.003096   | 0.001605 | 0.001738   | 0.003206    | 0.003868 |
| Jewel Changi         | 6000x4000  | 0.001546  | 0.001524   | 0.000704 | 0.000765   | 0.001549    | 0.001018 |
| Lake Atitlan         | 5112x3408  | 0.008325  | 0.006668   | 0.004817 | 0.004775   | 0.009906    | 0.006338 |
| Louvre               | 6056x4000  | 0.003376  | 0.004164   | 0.00155  | 0.001554   | 0.003499    | 0.002503 |
| Rosetta Nebula       | 5828x3885  | 0.005964  | 0.006424   | 0.002605 | 0.002859   | 0.003657    | 0.002781 |
| Termas Geometricas   | 5472x3648  | 0.003261  | 0.003039   | 0.001369 | 0.001563   | 0.003203    | 0.002181 |
| Yosemite Tunnel View | 5580x3720  | 0.003705  | 0.003239   | 0.001815 | 0.001703   | 0.004335    | 0.002751 |
|                      |            |           |            |          |            |             |          |
| Akihabara            | 1920x1278  | 0.005004  | 0.003252   | 0.002484 | 0.002534   | 0.004469    | 0.003833 |
| Boothbay             | 1920x1280  | 0.005954  | 0.005284   | 0.002996 | 0.002876   | 0.004831    | 0.004727 |
| Bryggen              | 1920x1080  | 0.002994  | 0.005939   | 0.001712 | 0.001426   | 0.002588    | 0.001812 |
| Futuristic           | 1920x1280  | 0.002313  | 0.002678   | 0.000846 | 0.000545   | 0.000671    | 0.000682 |
| Hokkaido             | 1920x1280  | 0.003056  | 0.002787   | 0.001492 | 0.001677   | 0.002862    | 0.003837 |
| Jewel Changi         | 1920x1280  | 0.001762  | 0.001543   | 0.000867 | 0.000954   | 0.001853    | 0.001411 |
| Lake Atitlan         | 1920x1280  | 0.00748   | 0.006809   | 0.004505 | 0.003943   | 0.007286    | 0.00592  |
| Louvre               | 1920x1268  | 0.003159  | 0.004407   | 0.001396 | 0.00123    | 0.002008    | 0.002204 |
| Rosetta Nebula       | 1920x1280  | 0.003934  | 0.004352   | 0.001557 | 0.001699   | 0.002179    | 0.001836 |
| Termas Geometricas   | 1920x1280  | 0.002634  | 0.0025     | 0.001151 | 0.001273   | 0.002145    | 0.00182  |
| Yosemite Tunnel View | 1920x1280  | 0.002827  | 0.003839   | 0.00165  | 0.001488   | 0.002599    | 0.002346 |
# With Dithering

## Time

| Image                | Dimensions | Wu (Oklab) | k-means | imagequant | exoquant |
|:-------------------- |:---------- | ----------:| -------:| ----------:| --------:|
| Akihabara            | 5663x3769  | 137        | 282     | 1303       | 7400     |
| Boothbay             | 6720x4480  | 190        | 392     | 1031       | 8921     |
| Bryggen              | 5508x3098  | 113        | 229     | 692        | 4549     |
| Futuristic           | 6000x4000  | 156        | 318     | 360        | 5874     |
| Hokkaido             | 6000x4000  | 157        | 317     | 786        | 7362     |
| Jewel Changi         | 6000x4000  | 152        | 316     | 592        | 6352     |
| Lake Atitlan         | 5112x3408  | 113        | 231     | 1182       | 6136     |
| Louvre               | 6056x4000  | 154        | 317     | 667        | 7020     |
| Rosetta Nebula       | 5828x3885  | 146        | 300     | 506        | 6494     |
| Termas Geometricas   | 5472x3648  | 127        | 263     | 1011       | 6201     |
| Yosemite Tunnel View | 5580x3720  | 133        | 273     | 1139       | 6709     |
|                      |            |            |         |            |          |
| Akihabara            | 1920x1278  | 17         | 38      | 319        | 1256     |
| Boothbay             | 1920x1280  | 17         | 38      | 227        | 1010     |
| Bryggen              | 1920x1080  | 14         | 33      | 149        | 658      |
| Futuristic           | 1920x1280  | 17         | 37      | 123        | 655      |
| Hokkaido             | 1920x1280  | 17         | 37      | 201        | 972      |
| Jewel Changi         | 1920x1280  | 17         | 37      | 166        | 790      |
| Lake Atitlan         | 1920x1280  | 17         | 38      | 302        | 1194     |
| Louvre               | 1920x1268  | 17         | 37      | 195        | 883      |
| Rosetta Nebula       | 1920x1280  | 17         | 38      | 157        | 767      |
| Termas Geometricas   | 1920x1280  | 17         | 37      | 215        | 959      |
| Yosemite Tunnel View | 1920x1280  | 17         | 37      | 212        | 933      |

## DSSIM

| Image                | Dimensions | Wu (Oklab) | k-means  | imagequant | exoquant |
|:-------------------- |:---------- |:---------- |:-------- |:---------- |:-------- |
| Akihabara            | 5663x3769  | 0.002581   | 0.002029 | 0.002289   | 0.00279  |
| Boothbay             | 6720x4480  | 0.00153    | 0.001303 | 0.001429   | 0.001843 |
| Bryggen              | 5508x3098  | 0.001504   | 0.000742 | 0.000996   | 0.00101  |
| Futuristic           | 6000x4000  | 0.000711   | 0.000405 | 0.000487   | 0.000471 |
| Hokkaido             | 6000x4000  | 0.001325   | 0.000969 | 0.001225   | 0.001961 |
| Jewel Changi         | 6000x4000  | 0.000899   | 0.000515 | 0.000581   | 0.001018 |
| Lake Atitlan         | 5112x3408  | 0.002805   | 0.002114 | 0.002039   | 0.002739 |
| Louvre               | 6056x4000  | 0.001156   | 0.000907 | 0.001063   | 0.001687 |
| Rosetta Nebula       | 5828x3885  | 0.001231   | 0.00096  | 0.001489   | 0.001311 |
| Termas Geometricas   | 5472x3648  | 0.001281   | 0.00093  | 0.001072   | 0.001452 |
| Yosemite Tunnel View | 5580x3720  | 0.001166   | 0.000935 | 0.000969   | 0.00156  |
|                      |            |           |            |         |            |
| Akihabara            | 1920x1278  | 0.001811   | 0.001503 | 0.001767   | 0.002141 |
| Boothbay             | 1920x1280  | 0.001833   | 0.001437 | 0.001692   | 0.001965 |
| Bryggen              | 1920x1080  | 0.001421   | 0.000659 | 0.000942   | 0.001009 |
| Futuristic           | 1920x1280  | 0.000663   | 0.000443 | 0.00046    | 0.000427 |
| Hokkaido             | 1920x1280  | 0.001252   | 0.001013 | 0.001474   | 0.002332 |
| Jewel Changi         | 1920x1280  | 0.000857   | 0.000562 | 0.000748   | 0.001277 |
| Lake Atitlan         | 1920x1280  | 0.002529   | 0.001854 | 0.001925   | 0.002298 |
| Louvre               | 1920x1268  | 0.001095   | 0.000788 | 0.001053   | 0.001537 |
| Rosetta Nebula       | 1920x1280  | 0.001061   | 0.000825 | 0.001316   | 0.001208 |
| Termas Geometricas   | 1920x1280  | 0.001115   | 0.000779 | 0.001071   | 0.001213 |
| Yosemite Tunnel View | 1920x1280  | 0.001072   | 0.000762 | 0.000977   | 0.001168 |

# Images with Transparency

The results below compare `quantette`'s `RgbaPipeline` against other libraries that support an alpha channel. The images used can be found in [img/alpha/](../img/alpha/), which includes photo cutouts, illustrations, logos, 3D renders, emoji, and images with smooth alpha fades.

The settings and libraries being compared are:

- `Wu`: `RgbaPipeline` using `RgbaQuantizeMethod::Wu`.
- `k-means`: `RgbaPipeline` using `RgbaQuantizeMethod::kmeans()` (the default).
- `imagequant` version `4.4.1` run with the default library options.
- `color_quant` version `1.1.0` (NeuQuant) run with a `sample_frac` of `1`.
- `exoquant` version `0.2.0` run without k-means optimization.

The "DSSIM" tables list the DSSIM values as reported by the `report-alpha` command of the `accuracy` binary found in [`examples/`](../examples/), which compares images with their alpha channel. The "Time" tables provide the total time, in milliseconds, for quantization and remapping as reported by the `cli` binary in [`examples/`](../examples/) with the `--verbose` and (for `quantette`) `--alpha` flags, using 4 threads. Each time is the minimum of 10 runs on an Apple M2 Max.

All tables below are for 256 colors. The summary table below covers 16, 64, and 256 colors, comparing each `quantette` method against `imagequant` across the 15 images:

| Dithering         | Colors | Method  | Images with lower DSSIM than imagequant | Mean DSSIM relative to imagequant |
|:------------------|:-------|:--------|:----------------------------------------|:----------------------------------|
| Without dithering | 16     | Wu      | 10 / 15                                 | 0.97                              |
| Without dithering | 16     | k-means | 13 / 15                                 | 0.90                              |
| Without dithering | 64     | Wu      | 8 / 15                                  | 1.10                              |
| Without dithering | 64     | k-means | 14 / 15                                 | 0.86                              |
| Without dithering | 256    | Wu      | 6 / 15                                  | 2.10                              |
| Without dithering | 256    | k-means | 13 / 15                                 | 0.90                              |
| With dithering    | 16     | Wu      | 6 / 15                                  | 1.02                              |
| With dithering    | 16     | k-means | 11 / 15                                 | 0.90                              |
| With dithering    | 64     | Wu      | 8 / 15                                  | 1.06                              |
| With dithering    | 64     | k-means | 14 / 15                                 | 0.82                              |
| With dithering    | 256    | Wu      | 7 / 15                                  | 1.64                              |
| With dithering    | 256    | k-means | 14 / 15                                 | 0.78                              |

Note that for generated palettes, `RgbaPipeline` always maps fully opaque pixels to fully opaque palette colors and fully transparent pixels to fully transparent palette colors (even with dithering), so opaque regions of an image stay opaque and transparent regions stay transparent. The other libraries do not necessarily guarantee this.

## Without Dithering

### Time

| Image               | Dimensions | Wu  | k-means | imagequant |
|:--------------------|:-----------|:----|:--------|:-----------|
| brown_rat           | 1151x1093  | 23  | 31      | 19         |
| carnival_mask       | 1612x2400  | 129 | 166     | 237        |
| cosmic_vista_fade   | 768x512    | 56  | 111     | 113        |
| crystal_ball        | 256x256    | 5   | 32      | 7          |
| cygnus_patch        | 1960x1960  | 71  | 110     | 66         |
| dna                 | 800x600    | 11  | 40      | 13         |
| fire                | 256x256    | 5   | 35      | 9          |
| fruits_vignette     | 768x512    | 54  | 106     | 101        |
| ghost               | 256x256    | 5   | 27      | 7          |
| glowing_star        | 256x256    | 7   | 21      | 6          |
| jellyfish           | 256x256    | 6   | 36      | 9          |
| king_penguin_cutout | 768x512    | 42  | 115     | 56         |
| owls                | 622x527    | 38  | 54      | 40         |
| protein             | 1098x1048  | 31  | 52      | 50         |
| rainbow             | 256x256    | 4   | 25      | 5          |

### DSSIM

| Image               | Dimensions | Wu       | k-means  | imagequant | color_quant | exoquant |
|:--------------------|:-----------|:---------|:---------|:-----------|:------------|:---------|
| brown_rat           | 1151x1093  | 0.000066 | 0.000004 | 0.000005   | 0.000012    | 0.00037  |
| carnival_mask       | 1612x2400  | 0.001549 | 0.001531 | 0.001556   | 0.002235    | 0.003177 |
| cosmic_vista_fade   | 768x512    | 0.003234 | 0.003178 | 0.004087   | 0.007856    | 0.006068 |
| crystal_ball        | 256x256    | 0.001289 | 0.000305 | 0.000335   | 0.000665    | 0.001    |
| cygnus_patch        | 1960x1960  | 0.000009 | 0.00001  | 0.000009   | 0.000017    | 0.000314 |
| dna                 | 800x600    | 0.000029 | 0.000026 | 0.000022   | 0.000083    | 0.000043 |
| fire                | 256x256    | 0.001773 | 0.000299 | 0.000325   | 0.000677    | 0.001022 |
| fruits_vignette     | 768x512    | 0.006812 | 0.00675  | 0.00806    | 0.015141    | 0.012409 |
| ghost               | 256x256    | 0.001539 | 0.000225 | 0.000256   | 0.000519    | 0.001466 |
| glowing_star        | 256x256    | 0.000436 | 0.000048 | 0.000052   | 0.000181    | 0.00046  |
| jellyfish           | 256x256    | 0.000484 | 0.00022  | 0.000223   | 0.000426    | 0.000712 |
| king_penguin_cutout | 768x512    | 0.001337 | 0.001093 | 0.001173   | 0.003082    | 0.004443 |
| owls                | 622x527    | 0.000436 | 0.000426 | 0.000496   | 0.000834    | 0.000868 |
| protein             | 1098x1048  | 0.00109  | 0.001099 | 0.001788   | 0.003033    | 0.003033 |
| rainbow             | 256x256    | 0.000586 | 0.000135 | 0.000139   | 0.000486    | 0.001053 |

## With Dithering

### Time

| Image               | Dimensions | Wu  | k-means | imagequant |
|:--------------------|:-----------|:----|:--------|:-----------|
| brown_rat           | 1151x1093  | 97  | 75      | 25         |
| carnival_mask       | 1612x2400  | 254 | 294     | 306        |
| cosmic_vista_fade   | 768x512    | 74  | 135     | 130        |
| crystal_ball        | 256x256    | 13  | 37      | 8          |
| cygnus_patch        | 1960x1960  | 207 | 244     | 79         |
| dna                 | 800x600    | 32  | 58      | 16         |
| fire                | 256x256    | 14  | 45      | 11         |
| fruits_vignette     | 768x512    | 72  | 124     | 120        |
| ghost               | 256x256    | 14  | 36      | 8          |
| glowing_star        | 256x256    | 15  | 32      | 7          |
| jellyfish           | 256x256    | 13  | 37      | 9          |
| king_penguin_cutout | 768x512    | 46  | 82      | 56         |
| owls                | 622x527    | 33  | 57      | 47         |
| protein             | 1098x1048  | 71  | 93      | 74         |
| rainbow             | 256x256    | 13  | 30      | 6          |

### DSSIM

| Image               | Dimensions | Wu       | k-means  | imagequant | exoquant |
|:--------------------|:-----------|:---------|:---------|:-----------|:---------|
| brown_rat           | 1151x1093  | 0.000097 | 0.000004 | 0.000005   | 0.000526 |
| carnival_mask       | 1612x2400  | 0.001066 | 0.001046 | 0.001298   | 0.002506 |
| cosmic_vista_fade   | 768x512    | 0.002192 | 0.00227  | 0.003244   | 0.004491 |
| crystal_ball        | 256x256    | 0.00062  | 0.000205 | 0.000285   | 0.000657 |
| cygnus_patch        | 1960x1960  | 0.000009 | 0.000009 | 0.000009   | 0.002109 |
| dna                 | 800x600    | 0.000027 | 0.000024 | 0.000022   | 0.000068 |
| fire                | 256x256    | 0.000775 | 0.000181 | 0.000244   | 0.000849 |
| fruits_vignette     | 768x512    | 0.003661 | 0.003435 | 0.004796   | 0.006627 |
| ghost               | 256x256    | 0.000807 | 0.000145 | 0.000216   | 0.001249 |
| glowing_star        | 256x256    | 0.000243 | 0.000041 | 0.000052   | 0.001385 |
| jellyfish           | 256x256    | 0.000271 | 0.000154 | 0.000177   | 0.000595 |
| king_penguin_cutout | 768x512    | 0.000845 | 0.000783 | 0.000894   | 0.003914 |
| owls                | 622x527    | 0.000353 | 0.000332 | 0.000483   | 0.000791 |
| protein             | 1098x1048  | 0.000713 | 0.000743 | 0.00115    | 0.002255 |
| rainbow             | 256x256    | 0.000281 | 0.000094 | 0.000132   | 0.001509 |
