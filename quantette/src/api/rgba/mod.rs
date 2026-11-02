#[cfg(feature = "threads")]
use crate::color_map::NearestNeighborParallelColorMap;
#[cfg(feature = "kmeans")]
use crate::kmeans::KmeansOptions;
use crate::{
    BoundedSlice, CreatePaletteBufError, ImageBuf, ImageRef, IndexedImage, IndexedImageCounts,
    LengthOutOfRange, PaletteBuf, PaletteSize, color_map::NearestNeighborColorMap,
    dither::FloydSteinberg,
};
use alloc::vec::Vec;
use generate::Class;
use palette::Srgba;
use space::{Composites, srgba8_to_composites};

/// Evaluates `$par` if `$parallel` is `true` and the `threads` feature is enabled.
/// Otherwise, evaluates `$serial`.
macro_rules! par_or_serial {
    ($parallel: expr, $par: expr, $serial: expr $(,)?) => {{
        #[cfg(feature = "threads")]
        let result = if $parallel { $par } else { $serial };
        #[cfg(not(feature = "threads"))]
        let result = {
            let _: bool = $parallel;
            $serial
        };
        result
    }};
}

mod dedup;
mod generate;
mod space;

/// The set of supported color quantization methods for [`RgbaPipeline`].
///
/// This mirrors [`QuantizeMethod`](crate::QuantizeMethod), but for colors with an alpha channel.
/// If the `kmeans` feature is enabled, then support will be added for that method.
/// Otherwise, only Wu's color quantization method and custom palettes are supported.
#[derive(Debug, Clone, PartialEq)]
pub enum RgbaQuantizeMethod {
    /// Quantize an image into a custom color palette.
    ///
    /// This method will simply map each pixel to its closest color in the palette.
    /// The palette is used as is, so it is not deduplicated.
    CustomPalette(PaletteBuf<Srgba<u8>>),
    /// Wu's color quantizer (Greedy Orthogonal Bipartitioning) on a 4 dimensional projection of
    /// the colors, followed by a few iterations of k-means refinement.
    ///
    /// This method is quick, but it gives noticeably worse results than k-means, especially for
    /// large palettes.
    ///
    /// See the [`wu`](crate::wu) module for more details.
    Wu,
    #[cfg(feature = "kmeans")]
    /// Color quantization using k-means clustering.
    ///
    /// Initial centroids are chosen with greedy k-means++, and they are then refined with
    /// Lloyd's algorithm over all of the distinct colors in the input. Since the result depends
    /// on the initial centroids, this is done a few times for small palettes, and the palette with
    /// the lowest error is kept.
    ///
    /// This method is slower than Wu's color quantizer but gives more accurate results.
    ///
    /// Only the [`seed`](KmeansOptions::seed) of the [`KmeansOptions`] is used, which seeds the
    /// k-means++ initialization. The results are deterministic for a given seed.
    Kmeans(KmeansOptions),
}

impl RgbaQuantizeMethod {
    #[cfg(feature = "kmeans")]
    /// Create a new [`RgbaQuantizeMethod::Kmeans`] with the default [`KmeansOptions`].
    #[must_use]
    pub const fn kmeans() -> Self {
        Self::Kmeans(KmeansOptions::new())
    }
}

impl From<PaletteBuf<Srgba<u8>>> for RgbaQuantizeMethod {
    #[inline]
    fn from(palette: PaletteBuf<Srgba<u8>>) -> Self {
        Self::CustomPalette(palette)
    }
}

#[cfg(feature = "kmeans")]
impl From<KmeansOptions> for RgbaQuantizeMethod {
    #[inline]
    fn from(options: KmeansOptions) -> Self {
        Self::Kmeans(options)
    }
}

impl TryFrom<Vec<Srgba<u8>>> for RgbaQuantizeMethod {
    type Error = CreatePaletteBufError<Srgba<u8>>;

    #[inline]
    fn try_from(colors: Vec<Srgba<u8>>) -> Result<Self, Self::Error> {
        PaletteBuf::try_from(colors).map(Into::into)
    }
}

/// A builder struct to specify quantization options for images with transparency.
///
/// This is the [`Srgba<u8>`] counterpart to [`Pipeline`](crate::Pipeline), and it has a similar API.
///
/// Each color is compared by how it looks composited over both a black and a white background,
/// with both composites in the [`Oklab`](palette::Oklab) color space. This accounts for how visible
/// differences in alpha are (e.g., a faint shadow over a dark background), and it treats all fully
/// transparent colors as the same color.
///
/// For images, colors are also weighted by importance: pixels in smooth regions are weighted more
/// than pixels in noisy or detailed regions, since errors are more noticeable in smooth regions.
///
/// The output palette contains straight (unassociated) sRGB and alpha values. Additionally,
/// generated palettes (i.e., not [`RgbaQuantizeMethod::CustomPalette`]) uphold the following:
/// - Fully opaque pixels are always mapped to a fully opaque palette color, and fully transparent
///   pixels are always mapped to a fully transparent palette color, even with dithering.
///   So, opaque regions of an image stay opaque and transparent regions stay transparent.
/// - Fully transparent palette colors are always `Srgba::new(0, 0, 0, 0)`.
/// - The palette has no duplicate colors, so it may contain fewer colors than requested.
///
/// # Examples
///
/// ```
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// use quantette::{ImageBuf, RgbaPipeline, RgbaQuantizeMethod};
///
/// // let img = image::open("some image")?.into_rgba8();
/// let img = image::RgbaImage::new(256, 256);
/// let img = ImageBuf::try_from(img)?;
///
/// let pipeline = RgbaPipeline::new()
///     .palette_size(32u16.try_into()?)
///     .quantize_method(RgbaQuantizeMethod::kmeans());
///
/// let quantized = pipeline.clone().input_image(img.as_ref()).output_srgba8_image();
/// assert_eq!(img.dimensions(), quantized.dimensions());
///
/// let indexed = pipeline.input_image(img.as_ref()).output_srgba8_indexed_image();
/// assert_eq!(indexed.palette().len(), 1);
/// # Ok(())
/// # }
/// ```
#[must_use]
#[derive(Debug, Clone, PartialEq)]
pub struct RgbaPipeline {
    /// The number of colors to put in the palette.
    k: PaletteSize,
    /// The color quantization method to use.
    quantize_method: RgbaQuantizeMethod,
    /// The ditherer to use, if any.
    ditherer: Option<FloydSteinberg>,
    #[cfg(feature = "threads")]
    /// Whether or not to run the pipeline in parallel.
    parallel: bool,
}

impl RgbaPipeline {
    /// Create a new [`RgbaPipeline`] with default options.
    pub fn new() -> Self {
        Self {
            k: PaletteSize::MAX,
            #[cfg(feature = "kmeans")]
            quantize_method: RgbaQuantizeMethod::kmeans(),
            #[cfg(not(feature = "kmeans"))]
            quantize_method: RgbaQuantizeMethod::Wu,
            ditherer: Some(FloydSteinberg::new()),
            #[cfg(feature = "threads")]
            parallel: false,
        }
    }

    /// Sets the palette size which determines the (maximum) number of colors to have in the palette.
    ///
    /// The default palette size is [`PaletteSize::MAX`].
    pub fn palette_size(mut self, size: PaletteSize) -> Self {
        self.k = size;
        self
    }

    /// Sets the color quantization method to use.
    ///
    /// The default quantization method is [`RgbaQuantizeMethod::Kmeans`] if the `kmeans` feature
    /// is enabled, since it gives much better results for images with transparency. Otherwise,
    /// the default is [`RgbaQuantizeMethod::Wu`].
    pub fn quantize_method(mut self, quantize_method: impl Into<RgbaQuantizeMethod>) -> Self {
        self.quantize_method = quantize_method.into();
        self
    }

    #[cfg(feature = "threads")]
    /// Sets whether or not to run the pipeline in parallel across multiple threads.
    ///
    /// Use a [`rayon::ThreadPool`] to specify the number of threads if necessary.
    ///
    /// By default, this is `false`.
    pub fn parallel(mut self, parallel: bool) -> Self {
        self.parallel = parallel;
        self
    }

    /// Sets the ditherer to use when quantizing an image, or disables dithering with `None`.
    ///
    /// Error diffusion applies to all four dimensions, including alpha.
    ///
    /// The default ditherer is [`FloydSteinberg::new`].
    pub fn ditherer(mut self, ditherer: impl Into<Option<FloydSteinberg>>) -> Self {
        self.ditherer = ditherer.into();
        self
    }

    /// Specify a slice of [`Srgba<u8>`] colors as input.
    ///
    /// Note that only palettes can be outputted when a slice is used as input.
    /// To get an image as an output, provide an image as input using [`input_image`](Self::input_image).
    ///
    /// # Errors
    ///
    /// Returns an error if `colors` is empty or has a length greater than [`MAX_PIXELS`](crate::MAX_PIXELS).
    #[inline]
    pub fn input_slice(
        self,
        colors: &[Srgba<u8>],
    ) -> Result<RgbaPipelineWithSliceInput<'_>, LengthOutOfRange> {
        let colors = BoundedSlice::new(colors)?;
        Ok(RgbaPipelineWithSliceInput { options: self, colors })
    }

    /// Specify a sRGBA image reference as input.
    ///
    /// See the [`ImageRef`] docs for more information as well as how to create one.
    #[inline]
    pub fn input_image(self, image: ImageRef<'_, Srgba<u8>>) -> RgbaPipelineWithImageRefInput<'_> {
        RgbaPipelineWithImageRefInput { options: self, image }
    }

    /// Returns whether or not to run in parallel.
    #[inline]
    #[cfg_attr(
        not(feature = "threads"),
        expect(clippy::unused_self, reason = "used if the threads feature is enabled")
    )]
    const fn is_parallel(&self) -> bool {
        #[cfg(feature = "threads")]
        let parallel = self.parallel;
        #[cfg(not(feature = "threads"))]
        let parallel = false;
        parallel
    }

    /// Computes the palette for the prepared input colors.
    ///
    /// Also returns whether the palette was generated, as opposed to being a custom palette.
    fn palette(&self, input: &Prepared) -> (PaletteBuf<Srgba<u8>>, bool) {
        /// The maximum number of colors to use for most iterations of Lloyd's algorithm.
        const LLOYD_SAMPLES: usize = 65536;
        /// The number of Lloyd's algorithm iterations on all colors after refining on the sample.
        const FULL_ITERATIONS: usize = 2;

        if let RgbaQuantizeMethod::CustomPalette(palette) = &self.quantize_method {
            return (palette.clone(), false);
        }

        let parallel = self.is_parallel();
        let Prepared { colors, classes, weights, .. } = input;
        let sample = generate::Weighted::sample(colors, classes, weights, LLOYD_SAMPLES);

        let (mut centroids, centroid_classes) = match &self.quantize_method {
            #[cfg(feature = "kmeans")]
            RgbaQuantizeMethod::Kmeans(options) => {
                generate::kmeans_centroids(&sample, classes, self.k, options.get_seed(), parallel)
            }
            RgbaQuantizeMethod::Wu | RgbaQuantizeMethod::CustomPalette(_) => {
                generate::wu_centroids(colors, classes, weights, &sample, self.k, parallel)
            }
        };

        if sample.colors.len() < colors.len() {
            generate::refine_centroids(
                &mut centroids,
                &centroid_classes,
                colors,
                classes,
                weights,
                FULL_ITERATIONS,
                parallel,
            );
        }

        (
            generate::finalize_palette(&centroids, &centroid_classes),
            true,
        )
    }
}

impl Default for RgbaPipeline {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

/// An [`RgbaPipeline`] paired with some slice of pixels, ready to be computed into a palette.
#[must_use]
#[derive(Debug, Clone, PartialEq)]
pub struct RgbaPipelineWithSliceInput<'a> {
    /// The options to use when generating the palette.
    options: RgbaPipeline,
    /// The input slice of pixels.
    colors: &'a BoundedSlice<Srgba<u8>>,
}

impl RgbaPipelineWithSliceInput<'_> {
    /// Runs the pipeline and returns the computed [`PaletteBuf<Srgba<u8>>`].
    #[must_use]
    pub fn output_srgba8_palette(self) -> PaletteBuf<Srgba<u8>> {
        let Self { options, colors } = self;
        if let RgbaQuantizeMethod::CustomPalette(palette) = options.quantize_method {
            return palette;
        }
        let input = Prepared::new(colors, colors.length(), 1, false, options.is_parallel());
        options.palette(&input).0
    }
}

/// An [`RgbaPipeline`] paired with some image, ready to be quantized or computed into a palette.
#[must_use]
#[derive(Debug, Clone, PartialEq)]
pub struct RgbaPipelineWithImageRefInput<'a> {
    /// The options to use when generating the quantized image.
    options: RgbaPipeline,
    /// The input image.
    image: ImageRef<'a, Srgba<u8>>,
}

impl RgbaPipelineWithImageRefInput<'_> {
    /// Deduplicates and converts the input image, returning `None` if it is empty.
    fn prepare(&self) -> Option<Prepared> {
        let Self { options, image } = self;
        let (width, height) = image.dimensions();
        (!image.is_empty())
            .then(|| Prepared::new(image.as_slice(), width, height, true, options.is_parallel()))
    }

    /// Runs the pipeline and returns the computed [`PaletteBuf<Srgba<u8>>`].
    ///
    /// Returns `None` if the input image was empty.
    #[must_use]
    pub fn output_srgba8_palette(self) -> Option<PaletteBuf<Srgba<u8>>> {
        let input = self.prepare()?;
        Some(self.options.palette(&input).0)
    }

    /// Runs the pipeline and returns the quantized [`IndexedImage<Srgba<u8>>`].
    #[must_use]
    pub fn output_srgba8_indexed_image(self) -> IndexedImage<Srgba<u8>> {
        let Some(input) = self.prepare() else {
            return IndexedImage::default();
        };
        let (palette, generated) = self.options.palette(&input);
        remap(
            &input,
            palette,
            generated,
            self.options.ditherer,
            self.options.is_parallel(),
        )
    }

    /// Runs the pipeline and returns the quantized [`ImageBuf<Srgba<u8>>`].
    #[must_use]
    pub fn output_srgba8_image(self) -> ImageBuf<Srgba<u8>> {
        let parallel = self.options.is_parallel();
        let indexed = self.output_srgba8_indexed_image();
        par_or_serial!(parallel, indexed.to_image_par(), indexed.to_image())
    }
}

/// The deduplicated input colors, prepared for quantization.
struct Prepared {
    /// The deduplicated input.
    image: IndexedImageCounts<Srgba<u8>, u32>,
    /// The unique colors in the [`Composites`] color space.
    colors: Vec<Composites>,
    /// The alpha class of each unique color.
    classes: Vec<Class>,
    /// The importance weight of each unique color.
    weights: Vec<f64>,
}

impl Prepared {
    /// Deduplicates and converts the given pixels.
    ///
    /// If `is_image` is true, then pixels are weighted based on their neighbors (see
    /// [`importance_weights`]). Otherwise, each pixel has the same weight.
    fn new(pixels: &[Srgba<u8>], width: u32, height: u32, is_image: bool, parallel: bool) -> Self {
        let image = dedup::dedup_image(pixels, width, height, parallel);
        let colors = srgba8_to_composites(image.palette(), parallel);
        let classes = image.palette().iter().map(|c| Class::of(c.alpha)).collect();
        let weights = if is_image {
            importance_weights(&image, &colors, parallel)
        } else {
            image.counts().iter().copied().map(f64::from).collect()
        };
        Self { image, colors, classes, weights }
    }
}

/// Computes the total importance weight of each unique color in a deduplicated image.
///
/// Errors are more noticeable in smooth regions (e.g., as banding) than in noisy or detailed
/// regions. So, each pixel is weighted based on how much its lightness differs from its neighbors,
/// with smooth regions having a higher weight.
fn importance_weights(
    image: &IndexedImageCounts<Srgba<u8>, u32>,
    colors: &[Composites],
    parallel: bool,
) -> Vec<f64> {
    /// The lightness difference at which a pixel's weight is halved.
    const SCALE: f32 = 0.05;
    /// The minimum weight of a pixel.
    const MIN_WEIGHT: f32 = 0.05;

    let width = image.width() as usize;
    let indices = image.indices();
    let lightness = colors.iter().map(|c| c.lightness()).collect::<Vec<_>>();
    let at = |i: usize| lightness[indices[i] as usize];

    let pixel_weight = |i: usize| {
        let (x, l) = (i % width, at(i));
        let mut diff = 0.0f32;
        if x > 0 {
            diff = diff.max((l - at(i - 1)).abs());
        }
        if x + 1 < width {
            diff = diff.max((l - at(i + 1)).abs());
        }
        if i >= width {
            diff = diff.max((l - at(i - width)).abs());
        }
        if i + width < indices.len() {
            diff = diff.max((l - at(i + width)).abs());
        }
        let d = diff / SCALE;
        (1.0 / (1.0 + d * d)).max(MIN_WEIGHT)
    };

    let pixel_weights = par_or_serial!(
        parallel,
        {
            use rayon::prelude::*;
            (0..indices.len())
                .into_par_iter()
                .map(pixel_weight)
                .collect::<Vec<_>>()
        },
        (0..indices.len()).map(pixel_weight).collect::<Vec<_>>(),
    );

    let mut weights = alloc::vec![0.0f64; colors.len()];
    for (&i, w) in indices.iter().zip(pixel_weights) {
        weights[i as usize] += f64::from(w);
    }
    weights
}

/// Maps each pixel of the prepared input to a color in `palette`.
///
/// For `generated` palettes, the mapping is constrained via [`generate::assign_constrained`],
/// matching how the palette was optimized. Custom palettes use plain nearest neighbor mapping.
fn remap(
    input: &Prepared,
    palette: PaletteBuf<Srgba<u8>>,
    generated: bool,
    ditherer: Option<FloydSteinberg>,
    parallel: bool,
) -> IndexedImage<Srgba<u8>> {
    let Prepared { image, colors, classes, .. } = input;
    let (width, height) = image.dimensions();

    // Map using the exact colors that will be outputted so that any dithering error is accurate.
    let palette_colors = srgba8_to_composites(&palette, false);
    let palette_classes = palette
        .iter()
        .map(|c| {
            if generated {
                Class::of(c.alpha)
            } else {
                Class::Free
            }
        })
        .collect::<Vec<_>>();

    let Some(ditherer) = ditherer else {
        // Without dithering, only the unique colors need to be mapped.
        let color_indices = generate::assign_constrained(
            colors,
            classes,
            &palette_colors,
            &palette_classes,
            parallel,
        );
        let map = |&i: &u32| color_indices[i as usize];
        let indices = par_or_serial!(
            parallel,
            {
                use rayon::prelude::*;
                image.indices().par_iter().map(map).collect()
            },
            image.indices().iter().map(map).collect(),
        );
        return IndexedImage::new_unchecked(width, height, palette.into_vec(), indices);
    };

    let pixels = image.as_indexed_image().map_ref(|_| colors.clone());
    let pixels = par_or_serial!(parallel, pixels.to_image_par(), pixels.to_image());
    let palette_buf = PaletteBuf::new_unchecked(palette_colors.clone());
    let mut indexed = par_or_serial!(
        parallel,
        ditherer.dither_image_to_indexed_par(
            pixels.as_ref(),
            NearestNeighborParallelColorMap::new(palette_buf),
        ),
        ditherer
            .dither_image_to_indexed(pixels.as_ref(), NearestNeighborColorMap::new(palette_buf)),
    )
    .map(|_| palette.into_vec());

    let pixel_classes = image.indices().iter().map(|&i| classes[i as usize]);
    undo_alpha_dither_leaks(
        pixels.as_slice(),
        pixel_classes,
        &mut indexed,
        &palette_colors,
        generated,
    );
    indexed
}

/// Error diffusion can map fully transparent or opaque pixels to palette colors with different
/// alpha values, which speckles transparent backgrounds and opaque regions. This undoes such
/// mappings.
///
/// For `generated` palettes, the same constraints as [`generate::assign_constrained`] are applied.
/// Otherwise, a pixel is only remapped if its nearest palette color has the same alpha as it.
fn undo_alpha_dither_leaks(
    pixels: &[Composites],
    pixel_classes: impl Iterator<Item = Class>,
    indexed: &mut IndexedImage<Srgba<u8>>,
    palette: &[Composites],
    generated: bool,
) {
    let palette_classes = indexed
        .palette()
        .iter()
        .map(|c| Class::of(c.alpha))
        .collect::<Vec<_>>();
    let has_class = |class| palette_classes.contains(&class);
    let (has_opaque, has_transparent) = (has_class(Class::Opaque), has_class(Class::Transparent));

    for ((index, &pixel), class) in indexed
        .indices_mut()
        .iter_mut()
        .zip(pixels)
        .zip(pixel_classes)
    {
        let applicable = match class {
            Class::Opaque => has_opaque,
            Class::Transparent => has_transparent,
            Class::Free => false,
        };
        if !applicable || palette_classes[usize::from(*index)] == class {
            continue;
        }
        let nearest = if generated {
            generate::nearest_where(palette, pixel, |i| palette_classes[i] == class)
        } else {
            generate::nearest_where(palette, pixel, |_| true)
                .filter(|&i| palette_classes[usize::from(i)] == class)
        };
        if let Some(i) = nearest {
            *index = i;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    fn image(pixels: &[Srgba<u8>]) -> ImageRef<'_, Srgba<u8>> {
        let width = u32::try_from(pixels.len()).unwrap();
        ImageRef::new(width, 1, pixels).unwrap()
    }

    fn methods() -> Vec<RgbaQuantizeMethod> {
        vec![
            RgbaQuantizeMethod::Wu,
            #[cfg(feature = "kmeans")]
            RgbaQuantizeMethod::kmeans(),
        ]
    }

    fn pipelines(k: u8) -> Vec<RgbaPipeline> {
        let mut pipelines = Vec::new();
        for method in methods() {
            for ditherer in [None, Some(FloydSteinberg::new())] {
                let pipeline = RgbaPipeline::new()
                    .palette_size(PaletteSize::from_u8_clamped(k))
                    .quantize_method(method.clone())
                    .ditherer(ditherer);
                #[cfg(feature = "threads")]
                pipelines.push(pipeline.clone().parallel(true));
                pipelines.push(pipeline);
            }
        }
        pipelines
    }

    /// An opaque gradient circle with an anti-aliased edge on a transparent background.
    pub(super) fn sprite() -> (u32, u32, Vec<Srgba<u8>>) {
        let (width, height) = (64u32, 64u32);
        let mut pixels = Vec::new();
        for y in 0..height {
            for x in 0..width {
                let dx = f32::from(u16::try_from(x).unwrap_or(0)) - 32.0;
                let dy = f32::from(u16::try_from(y).unwrap_or(0)) - 32.0;
                let coverage = (21.0 - libm::sqrtf(dx * dx + dy * dy)).clamp(0.0, 1.0);
                #[expect(
                    clippy::cast_possible_truncation,
                    clippy::cast_sign_loss,
                    reason = "test"
                )]
                let alpha = (coverage * 255.0).round() as u8;
                #[expect(clippy::cast_possible_truncation, reason = "test")]
                pixels.push(Srgba::new((x * 4) as u8, (y * 4) as u8, 128, alpha));
            }
        }
        (width, height, pixels)
    }

    #[test]
    fn opaque_stays_opaque_and_transparent_stays_transparent() {
        let (width, height, pixels) = sprite();
        let input = ImageRef::new(width, height, pixels.as_slice()).unwrap();
        for k in [2, 4, 16, 255] {
            for pipeline in pipelines(k) {
                let output = pipeline.clone().input_image(input).output_srgba8_image();
                for (i, o) in pixels.iter().zip(output.as_slice()) {
                    if i.alpha == u8::MAX {
                        assert_eq!(o.alpha, u8::MAX, "{pipeline:?}");
                    }
                    if i.alpha == 0 {
                        assert_eq!(*o, Srgba::new(0, 0, 0, 0), "{pipeline:?}");
                    }
                }
            }
        }
    }

    #[test]
    fn exact_colors_are_preserved() {
        let pixels = [
            Srgba::new(255, 0, 0, 255),
            Srgba::new(0, 0, 255, 128),
            Srgba::new(0, 255, 0, 64),
            Srgba::new(0, 0, 0, 0),
        ];
        for pipeline in pipelines(4) {
            let output = pipeline
                .clone()
                .input_image(image(&pixels))
                .output_srgba8_image();
            assert_eq!(output.as_slice(), &pixels, "{pipeline:?}");
        }
    }

    #[test]
    fn fully_transparent_colors_are_equivalent() {
        let pixels = [Srgba::new(255, 0, 0, 0), Srgba::new(0, 255, 255, 0)];
        for pipeline in pipelines(2) {
            let indexed = pipeline
                .clone()
                .input_image(image(&pixels))
                .output_srgba8_indexed_image();
            assert_eq!(indexed.palette(), &[Srgba::new(0, 0, 0, 0)], "{pipeline:?}");
            assert_eq!(indexed.indices(), &[0, 0], "{pipeline:?}");
        }
    }

    #[test]
    fn palettes_are_bounded_and_deduplicated() {
        let (_, _, pixels) = sprite();
        for k in [1, 2, 7, 64, 255] {
            for pipeline in pipelines(k) {
                let palette = pipeline
                    .clone()
                    .input_slice(&pixels)
                    .unwrap()
                    .output_srgba8_palette();
                assert!(
                    !palette.is_empty() && palette.len() <= usize::from(k),
                    "{pipeline:?}"
                );
                for (i, a) in palette.iter().enumerate() {
                    assert!(!palette.as_slice()[..i].contains(a), "{pipeline:?}");
                    if a.alpha == 0 {
                        assert_eq!(*a, Srgba::new(0, 0, 0, 0));
                    }
                }
            }
        }
    }

    #[test]
    fn serial_output_is_deterministic() {
        let (width, height, pixels) = sprite();
        let input = ImageRef::new(width, height, pixels.as_slice()).unwrap();
        for pipeline in pipelines(16) {
            let first = pipeline
                .clone()
                .input_image(input)
                .output_srgba8_indexed_image();
            let second = pipeline.input_image(input).output_srgba8_indexed_image();
            assert_eq!(first, second);
        }
    }

    #[test]
    fn empty_inputs() {
        for pipeline in pipelines(16) {
            let input = ImageRef::<Srgba<u8>>::default();
            assert!(
                pipeline
                    .clone()
                    .input_image(input)
                    .output_srgba8_indexed_image()
                    .is_empty()
            );
            assert!(
                pipeline
                    .clone()
                    .input_image(input)
                    .output_srgba8_image()
                    .is_empty()
            );
            assert!(
                pipeline
                    .clone()
                    .input_image(input)
                    .output_srgba8_palette()
                    .is_none()
            );
            assert!(pipeline.input_slice(&[]).is_err());
        }
    }

    #[test]
    fn custom_palette_is_used_as_is() {
        let palette = vec![
            Srgba::new(0, 0, 0, 0),
            Srgba::new(255, 255, 255, 250),
            Srgba::new(0, 0, 0, 255),
        ];
        let pixels = [
            Srgba::new(10, 200, 30, 0),
            Srgba::new(250, 250, 250, 255),
            Srgba::new(5, 5, 5, 255),
        ];
        let pipeline = RgbaPipeline::new()
            .ditherer(None)
            .quantize_method(RgbaQuantizeMethod::try_from(palette.clone()).unwrap());
        let indexed = pipeline
            .clone()
            .input_image(image(&pixels))
            .output_srgba8_indexed_image();
        assert_eq!(indexed.palette(), palette.as_slice());
        assert_eq!(indexed.indices(), &[0, 1, 2]);
        let output = pipeline
            .input_slice(&pixels)
            .unwrap()
            .output_srgba8_palette();
        assert_eq!(output.as_slice(), palette.as_slice());
    }

    #[test]
    fn faint_alpha_levels_are_preserved() {
        // A shadow fading out over every alpha level.
        let pixels = (0..=255u8)
            .map(|a| Srgba::new(20, 20, 30, a))
            .collect::<Vec<_>>();
        for pipeline in pipelines(255) {
            let output = pipeline
                .clone()
                .input_image(image(&pixels))
                .output_srgba8_image();
            for (i, o) in pixels.iter().zip(output.as_slice()) {
                assert!(i.alpha.abs_diff(o.alpha) <= 1, "{pipeline:?} {i:?} {o:?}");
            }
        }
    }

    #[cfg(all(feature = "kmeans", feature = "threads"))]
    #[test]
    fn parallel_kmeans_matches_serial() {
        let (width, height, pixels) = sprite();
        let input = ImageRef::new(width, height, pixels.as_slice()).unwrap();
        for ditherer in [None, Some(FloydSteinberg::new())] {
            let pipeline = RgbaPipeline::new()
                .palette_size(PaletteSize::from_u8_clamped(16))
                .quantize_method(RgbaQuantizeMethod::kmeans())
                .ditherer(ditherer);
            let serial = pipeline
                .clone()
                .input_image(input)
                .output_srgba8_indexed_image();
            let parallel = pipeline
                .parallel(true)
                .input_image(input)
                .output_srgba8_indexed_image();
            assert_eq!(serial, parallel);
        }
    }
}
