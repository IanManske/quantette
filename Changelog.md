# v0.4.0

This release is a major overhaul of the code and API of `quantette`. The API has been slightly improved and simplified, necessitating breaking changes. Performance should be also be better across the board, varying from minor to significant speedups.

The lists below don't cover every single change, rather only the highlights. See the [docs.rs](https://docs.rs/quantette/latest/quantette/) page to see the new API, docs, and examples.

## New

- Custom palettes can now be provided instead of always generating one from the source image. See `QuantizeMethod::CustomPalette`.
- In theory, this library is now `no_std` compatible (still needs `alloc` though). The `image`, `threads`, and `std` features must be disabled.
- A new `color_space` module is available, containing SIMD accelerated conversions between Oklab and sRGB.
- New structs have been added for color mapping like `NearestNeighborColorMap` and `PaletteSubstitution`.

## Breaking

- The previous `ImagePipeline` and `PalettePipeline` builder structs have been merged into the new `Pipeline` struct. Also, the `pipelines` crate feature has been removed (`Pipeline` is always available).
- The `colorspaces` crate feature and pipeline option has been removed. All quantization done via the `Pipeline` struct is now performed in the Oklab color space. The underlying quantization methods still support sRGB and other color spaces.
- There is a new `KmeansOptions` struct that is required by all k-means quantization functions in the new `Kmeans` struct.
- The API for Wu's quantization method is now handled by two structs: `WuU8x3` and `WuF32x3`.
- Several new types have been added to clean up the overall API and enforce invariants:

  - `ImageBuf`, `ImageRef`, and `ImageMut`
  - `PaletteBuf`, `Palette`, and `PaletteCounts`
  - `IndexedImage` and `IndexedImageCounts`

  Old types like `QuantizationOutput`, `IndexedColorCounts`, and `UniqueColorCounts` have been removed in favor of the ones above.

- Similarly, the new `IndexedColorMap` trait has been added to take the place of the `ColorCountsRemap` and `ColorCountsParallelRemap` traits. All image remapping, including dithering, is now generic over an implementation of `IndexedColorMap`.

## Other

- MSRV is now 1.90.
- Fixed minimal dependency versions.
- Updated examples and benchmarks in [docs/](). Note that different hardware and `RUSTFLAGS` were used for the new benchmarks, and so it isn't comparable to previous results.

# v0.3.0

## Breaking
- Removed the option to specify initial centroids in the high-level builder API. This simplified the API and underlying code. If you want to specify initial centroids, then you must now use the low-level API (module functions).
- The high-level builder structs `ImagePipeline` and `PalettePipeline` are no longer consuming builders. Their various methods now take references to `self` instead of owned `self`s.

## Other
- Added some `PartialEq` and `Eq` derives to structs and enums.
- Made some more functions `const`.
- Eliminated some usages of `unsafe`.
- Fixed a segment of code that could potentially panic.
- Updated/expanded documentation.

# v0.2.0
- Add multi-threaded versions of the dither functions.
- Pixel deduplication through `UniqueColorCounts` and `IndexedColorCounts` should be slightly faster for small images.
- External crates that have types present in `quantette`'s public API are now reexported (`palette` and `image`).
- `PalettePipeline` and `ImagePipeline` now take `impl Into<PaletteSize>` instead of just `PaletteSize` for their `palette_size` functions.
- Similarly, the pipeline structs now take `impl Into<QuantizeMethod<_>>` instead of just `QuantizeMethod<_>` for their `quantize_method` functions.
- Bumped `image` version to `0.25.0`.
- Removed unused `wide` feature on `palette` dependency.

# v0.1.1
Fixed typos and reduced unnecessary dependencies.

# v0.1.0
First release!
