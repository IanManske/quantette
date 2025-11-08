use clap::{Args, Parser, Subcommand, ValueEnum};
use image::{RgbImage, RgbaImage, buffer::ConvertBuffer as _};
use palette::{Oklab, cast::IntoComponents as _};
use quantette::{
    Image, IndexedImageCounts, PaletteSize,
    color_map::{IndexedColorMap, PaletteSubstitution},
    color_space::{oklab_to_srgb8, srgb8_to_oklab_par},
    dedup,
    dither::FloydSteinberg,
    kmeans::{Kmeans, KmeansOptions},
    wu::{BinnerF32x3, WuF32x3},
};
use rgb::{FromSlice as _, RGB8, RGBA};
use std::{
    ffi::OsStr,
    fmt::{self, Debug, Display},
    path::PathBuf,
};

#[path = "../util/mod.rs"]
mod util;

/// Set of algorithm choices to create a palette
#[derive(Debug, Copy, Clone, ValueEnum)]
enum Algorithm {
    Minibatch,
    Online,
    Wu,
    Neuquant,
    Exoquant,
    Imagequant,
}

impl Display for Algorithm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}",
            match self {
                Algorithm::Minibatch => "minibatch",
                Algorithm::Online => "online",
                Algorithm::Wu => "wu",
                Algorithm::Neuquant => "neuquant",
                Algorithm::Imagequant => "imagequant",
                Algorithm::Exoquant => "exoquant",
            }
        )
    }
}

#[derive(Args)]
struct Report {
    #[arg(short, long, default_value_t = Algorithm::Wu)]
    algo: Algorithm,

    #[arg(short, long, default_value = "16,64,256", value_delimiter = ',', value_parser = parse_palette_size)]
    k: Vec<PaletteSize>,

    #[arg(long, default_value_t = KmeansOptions::new().get_sampling_factor())]
    sampling_factor: f32,

    #[arg(long, default_value_t = KmeansOptions::new().get_batch_size())]
    batch_size: u32,

    #[arg(long, default_value_t = 1)]
    sample_frac: u8,

    #[arg(long)]
    kmeans_optimize: bool,

    #[arg(long)]
    dither: bool,

    #[arg(long, default_value_t = FloydSteinberg::DEFAULT_ERROR_DIFFUSION)]
    dither_error_diffusion: f32,

    images: Vec<PathBuf>,
}

impl Report {
    fn kmeans_options(&self) -> KmeansOptions {
        let Self { batch_size, sampling_factor, .. } = *self;
        KmeansOptions::new()
            .sampling_factor(sampling_factor)
            .batch_size(batch_size)
    }
}

#[derive(Subcommand)]
enum Command {
    Report(Report),
    Compare { image_a: PathBuf, image_b: PathBuf },
}

#[derive(Parser)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

fn parse_palette_size(s: &str) -> Result<PaletteSize, String> {
    let value: u16 = s.parse().map_err(|e| format!("{e}"))?;
    value.try_into().map_err(|e| format!("{e}"))
}

const COL_WIDTH: usize = 10;
const NUM_DECIMALS: usize = 4;

fn main() {
    let Cli { command } = Cli::parse();

    match command {
        Command::Report(options) => report(&options),
        Command::Compare { image_a, image_b } => {
            let ds = dssim::new();
            let a = image::open(image_a).unwrap().into_rgb8();
            let b = image::open(image_b).unwrap().into_rgb8();
            let a = ds
                .create_image_rgb(a.as_rgb(), a.width() as usize, a.height() as usize)
                .unwrap();

            let b = ds
                .create_image_rgb(b.as_rgb(), b.width() as usize, b.height() as usize)
                .unwrap();

            println!("{}", f64::from(ds.compare(&a, &b).0))
        }
    }
}

#[allow(clippy::too_many_lines)]
fn report(options: &Report) {
    fn each_image<F1, F2>(
        options: &Report,
        images: Vec<(String, RgbImage)>,
        name_len: usize,
        mut f1: F1,
    ) where
        F1: FnMut(RgbImage) -> F2,
        F2: FnMut(PaletteSize) -> Vec<RGB8>,
    {
        let ds = dssim::new();
        for (path, image) in images {
            let width = image.width() as usize;
            let height = image.height() as usize;

            let original = ds.create_image_rgb(image.as_rgb(), width, height).unwrap();

            let mut f2 = f1(image);
            let ssim_by_k = options
                .k
                .iter()
                .map(|&k| {
                    let quantized = ds.create_image_rgb(&f2(k), width, height).unwrap();
                    let ssim = 100.0 * f64::from(ds.compare(&original, quantized).0);
                    format!("{ssim:>COL_WIDTH$.NUM_DECIMALS$}")
                })
                .collect::<Vec<_>>()
                .join(" ");

            println!("{path:name_len$} {ssim_by_k}");
        }
    }

    fn each_image_indexed_counts<Map: Debug + IndexedColorMap<Oklab, Output = Oklab>>(
        options: &Report,
        images: Vec<(String, RgbImage)>,
        name_len: usize,
        f: impl Fn(&IndexedImageCounts<Oklab, u32>, PaletteSize) -> Map + Copy,
    ) {
        let ditherer = options
            .dither
            .then(|| FloydSteinberg::with_error_diffusion(options.dither_error_diffusion).unwrap());

        each_image(options, images, name_len, |image| {
            let image = Image::try_from(image).unwrap();
            let image = dedup::dedup_image_u8_3_counts_par(image.as_ref())
                .map(|palette| srgb8_to_oklab_par(&palette));

            move |k| {
                let color_map = f(&image, k);
                let color_map = PaletteSubstitution::from_slice_mapping(color_map, oklab_to_srgb8);
                let image = if let Some(ditherer) = ditherer {
                    ditherer.dither_indexed_to_image(image.as_ref(), color_map)
                } else {
                    image.map_to_image_par(color_map)
                };
                let buf = image.into_inner().into_components();
                bytemuck::cast_vec(buf)
            }
        });
    }

    let images = if options.images.is_empty() {
        util::load_image_dir_relative_to_root(
            ["img", "CQ100", "img"].into_iter().collect::<PathBuf>(),
        )
        .into_iter()
        .map(|(path, img)| {
            let name = path.file_stem().and_then(OsStr::to_str).unwrap().to_owned();
            (name, img)
        })
        .collect::<Vec<_>>()
    } else {
        options
            .images
            .iter()
            .map(|path| path.display().to_string())
            .zip(util::load_images(&options.images))
            .collect()
    };

    // use char count as supplement for grapheme count
    let max_name_len = images
        .iter()
        .map(|(name, _)| name.chars().count())
        .max()
        .unwrap_or(0);

    println!(
        "{:max_name_len$} {}",
        "image",
        options
            .k
            .iter()
            .map(|k| format!(
                "{:>1$} {2}",
                k,
                COL_WIDTH - NUM_DECIMALS - 1,
                str::repeat(" ", NUM_DECIMALS)
            ))
            .collect::<Vec<_>>()
            .join(" "),
    );

    match options.algo {
        Algorithm::Minibatch => {
            each_image_indexed_counts(options, images, max_name_len, |image, k| {
                let centroids =
                    WuF32x3::run_indexed_image_counts_par(image, BinnerF32x3::oklab_from_srgb8())
                        .unwrap()
                        .palette(k);
                let options = options.kmeans_options();
                Kmeans::run_indexed_image_par(image.as_ref(), centroids, options)
                    .into_parallel_color_map()
            });
        }
        Algorithm::Online => {
            each_image_indexed_counts(options, images, max_name_len, |image, k| {
                let centroids =
                    WuF32x3::run_indexed_image_counts_par(image, BinnerF32x3::oklab_from_srgb8())
                        .unwrap()
                        .palette(k);
                let options = options.kmeans_options();
                Kmeans::run_indexed_image(image.as_ref(), centroids, options)
                    .into_parallel_color_map()
            });
        }
        Algorithm::Wu => {
            each_image_indexed_counts(options, images, max_name_len, |image, k| {
                WuF32x3::run_indexed_image_counts_par(image, BinnerF32x3::oklab_from_srgb8())
                    .unwrap()
                    .parallel_color_map(k)
            });
        }
        Algorithm::Neuquant => {
            each_image(options, images, max_name_len, |image| {
                let image: RgbaImage = image.convert();

                move |k| {
                    let nq =
                        color_quant::NeuQuant::new(options.sample_frac.into(), k.into(), &image);

                    let colors = nq
                        .color_map_rgba()
                        .as_rgba()
                        .iter()
                        .map(RGBA::rgb)
                        .collect::<Vec<_>>();

                    image
                        .chunks_exact(4)
                        .map(|pix| colors[nq.index_of(pix)])
                        .collect()
                }
            });
        }
        Algorithm::Imagequant => {
            each_image(options, images, max_name_len, |image| {
                let image: RgbaImage = image.convert();

                move |k| {
                    let mut libq = imagequant::new();

                    let mut img = libq
                        .new_image(
                            image.as_rgba(),
                            image.width() as usize,
                            image.height() as usize,
                            0.0,
                        )
                        .unwrap();

                    libq.set_max_colors(k.as_u16().into()).unwrap();

                    let mut quantized = libq.quantize(&mut img).unwrap();
                    if !options.dither {
                        quantized.set_dithering_level(0.0).unwrap()
                    }
                    let (colors, indices) = quantized.remapped(&mut img).unwrap();

                    indices
                        .into_iter()
                        .map(|i| colors[usize::from(i)].rgb())
                        .collect()
                }
            });
        }
        Algorithm::Exoquant => {
            use exoquant::{Color, convert_to_indexed, ditherer, optimizer};

            each_image(options, images, max_name_len, |image| {
                let pixels = image
                    .pixels()
                    .map(|p| Color::new(p.0[0], p.0[1], p.0[2], u8::MAX))
                    .collect::<Vec<_>>();

                move |k| {
                    let k = k.into();
                    let (colors, indices) = match (options.kmeans_optimize, options.dither) {
                        (true, true) => convert_to_indexed(
                            &pixels,
                            image.width() as usize,
                            k,
                            &optimizer::KMeans,
                            &ditherer::FloydSteinberg::new(),
                        ),
                        (true, false) => convert_to_indexed(
                            &pixels,
                            image.width() as usize,
                            k,
                            &optimizer::KMeans,
                            &ditherer::None,
                        ),
                        (false, true) => convert_to_indexed(
                            &pixels,
                            image.width() as usize,
                            k,
                            &optimizer::None,
                            &ditherer::FloydSteinberg::new(),
                        ),
                        (false, false) => convert_to_indexed(
                            &pixels,
                            image.width() as usize,
                            k,
                            &optimizer::None,
                            &ditherer::None,
                        ),
                    };

                    indices
                        .into_iter()
                        .map(|i| {
                            let color = colors[usize::from(i)];
                            RGB8::new(color.r, color.g, color.b)
                        })
                        .collect()
                }
            });
        }
    }
}
