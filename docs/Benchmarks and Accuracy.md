# Benchmarks and Accuracy

Below are some comparisons between `quantette` and some other libraries. Each library has different situations and/or options that can make then perform better or worse, but these limited comparisons should give a rough point of reference. Note that `quantette` currently doesn't support alpha channel/component, while the other libraries do.

# Setup

The settings and libraries being compared are:

- `Wu (sRGB)`: `quantette`'s Wu's color quantizer implementation running in the sRGB color space. The default number of bins was used.
- `Wu (Oklab)`: `quantette`'s Wu's color quantizer implementation running in the Oklab color space. The default number of bins was used.
- `k-means`: `quantette`'s k-means color quantizer running in the Oklab color space. The default `KmeansOptions` were used.
- `imagequant` version `4.4.1` run with the default library options (quality of `100`).
- `color_quant` version `1.1.0` run with a `sample_frac` of `10`.
- `exoquant` version `0.2.0` run without k-means optimization, since it would otherwise take way too long.

The "Time" tables below provide the total time, in milliseconds, for quantization and remapping as reported by the `cli` binary in [`examples/`](../examples/) with the `--verbose` flag. 30 trials were run and averaged for each data point. The `Wu`, `k-means`, and `imagequant` columns used 4 threads, while `color_quant` and `exoquant` only support single-threaded execution. So, multiply or divide by 4 as you see fit.

The "Accuracy" tables list the DSSIM values as reported by the `accuracy` binary found in [`examples/`](../examples/) using the [`dssim`](https://crates.io/crates/dssim) crate. Note that `exoquant` results are not deterministic, since it uses `rand::random()`.

`color_quant` does not have dithering, so it is not included in the `With Dithering` tables below. Also, `Wu (sRGB)` does not easily support dithering and is omitted from the dithering tables below.

All results below are for 256 colors, and the `-C target-feature=+avx2` `rustc` flags were used. The images used can be found in [img/unsplash/img/Original](../img/unsplash/img/Original), and they are all roughly 6000x4000 in resolution.

# Without Dithering

## Time

| Image                    | Wu (sRGB) | Wu (Oklab) | k-means | imagequant | color_quant | exoquant |
| ------------------------ | --------- | ---------- | ------- | ---------- | ----------- | -------- |
| Akihabara.jpg            | 25        | 80         | 97      | 1044       | 2155        | 4815     |
| Boothbay.jpg             | 34        | 83         | 91      | 694        | 2801        | 4836     |
| Bryggen.jpg              | 21        | 46         | 53      | 530        | 1122        | 2155     |
| Futuristic.jpg           | 30        | 72         | 76      | 253        | 1572        | 2598     |
| Hokkaido.jpg             | 31        | 66         | 74      | 513        | 2064        | 3652     |
| Jewel Changi.jpg         | 28        | 69         | 75      | 442        | 1733        | 3156     |
| Lake Atitlan.jpg         | 21        | 64         | 78      | 959        | 1686        | 3840     |
| Louvre.jpg               | 29        | 68         | 76      | 474        | 2201        | 3438     |
| Rosetta Nebula.jpg       | 28        | 60         | 67      | 319        | 1900        | 2839     |
| Termas Geometricas.jpg   | 23        | 57         | 66      | 825        | 2002        | 3456     |
| Yosemite Tunnel View.jpg | 24        | 57         | 64      | 899        | 2104        | 3759     |

## Accuracy/DSSIM

| Image                    | Wu (sRGB) | Wu (Oklab) | k-means  | imagequant | color_quant | exoquant |
| ------------------------ | --------- | ---------- | -------- | ---------- | ----------- | -------- |
| Akihabara.jpg            | 0.007178  | 0.005379   | 0.003991 | 0.004436   | 0.007313    | 0.005661 |
| Boothbay.jpg             | 0.004569  | 0.004305   | 0.002464 | 0.002419   | 0.005882    | 0.003464 |
| Bryggen.jpg              | 0.003877  | 0.00618    | 0.002001 | 0.00207    | 0.003315    | 0.002234 |
| Futuristic.jpg           | 0.001906  | 0.002126   | 0.000658 | 0.000608   | 0.001026    | 0.000663 |
| Hokkaido.jpg             | 0.003467  | 0.003096   | 0.001605 | 0.001738   | 0.003206    | 0.003868 |
| Jewel Changi.jpg         | 0.001546  | 0.001524   | 0.000704 | 0.000765   | 0.001549    | 0.001018 |
| Lake Atitlan.jpg         | 0.008325  | 0.006668   | 0.004817 | 0.004775   | 0.009906    | 0.006338 |
| Louvre.jpg               | 0.003376  | 0.004164   | 0.00155  | 0.001554   | 0.003499    | 0.002503 |
| Rosetta Nebula.jpg       | 0.005964  | 0.006424   | 0.002605 | 0.002859   | 0.003657    | 0.002781 |
| Termas Geometricas.jpg   | 0.003261  | 0.003039   | 0.001369 | 0.001563   | 0.003203    | 0.002181 |
| Yosemite Tunnel View.jpg | 0.003705  | 0.003239   | 0.001815 | 0.001703   | 0.004335    | 0.002751 |

# With Dithering

## Time

| Image                    | Wu (Oklab) | k-means | imagequant | exoquant |
| ------------------------ | ---------- | ------- | ---------- | -------- |
| Akihabara.jpg            | 137        | 282     | 1303       | 7400     |
| Boothbay.jpg             | 190        | 392     | 1031       | 8921     |
| Bryggen.jpg              | 113        | 229     | 692        | 4549     |
| Futuristic.jpg           | 156        | 318     | 360        | 5874     |
| Hokkaido.jpg             | 157        | 317     | 786        | 7362     |
| Jewel Changi.jpg         | 152        | 316     | 592        | 6352     |
| Lake Atitlan.jpg         | 113        | 231     | 1182       | 6136     |
| Louvre.jpg               | 154        | 317     | 667        | 7020     |
| Rosetta Nebula.jpg       | 146        | 300     | 506        | 6494     |
| Termas Geometricas.jpg   | 127        | 263     | 1011       | 6201     |
| Yosemite Tunnel View.jpg | 133        | 273     | 1139       | 6709     |

## Accuracy/DSSIM

| Image                    | Wu (Oklab) | k-means  | imagequant | exoquant |
| ------------------------ | ---------- | -------- | ---------- | -------- |
| Akihabara.jpg            | 0.002581   | 0.002029 | 0.002289   | 0.00279  |
| Boothbay.jpg             | 0.00153    | 0.001303 | 0.001429   | 0.001843 |
| Bryggen.jpg              | 0.001504   | 0.000742 | 0.000996   | 0.00101  |
| Futuristic.jpg           | 0.000711   | 0.000405 | 0.000487   | 0.000471 |
| Hokkaido.jpg             | 0.001325   | 0.000969 | 0.001225   | 0.001961 |
| Jewel Changi.jpg         | 0.000899   | 0.000515 | 0.000581   | 0.001018 |
| Lake Atitlan.jpg         | 0.002805   | 0.002114 | 0.002039   | 0.002739 |
| Louvre.jpg               | 0.001156   | 0.000907 | 0.001063   | 0.001687 |
| Rosetta Nebula.jpg       | 0.001231   | 0.00096  | 0.001489   | 0.001311 |
| Termas Geometricas.jpg   | 0.001281   | 0.00093  | 0.001072   | 0.001452 |
| Yosemite Tunnel View.jpg | 0.001166   | 0.000935 | 0.000969   | 0.00156  |

