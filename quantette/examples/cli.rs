use clap::{Parser, Subcommand};
use image::RgbImage;
use palette::Srgb;
use quantette::{
    Image, IndexedImage, PaletteSize, Pipeline, QuantizeMethod,
    dither::FloydSteinberg,
    kmeans::KmeansOptions,
    wu::{BinnerU8x3, WuU8x3},
};
use rgb::FromSlice as _;
use std::path::PathBuf;

#[derive(Subcommand)]
enum Quantizer {
    Quantette {
        #[arg(long)]
        dedup: Option<bool>,

        #[arg(long)]
        srgb: bool,

        #[arg(long)]
        kmeans: bool,

        #[arg(long)]
        dither: bool,

        #[arg(long, default_value_t = FloydSteinberg::DEFAULT_ERROR_DIFFUSION)]
        dither_error_diffusion: f32,

        #[arg(long, default_value_t = KmeansOptions::new().get_sampling_factor())]
        sampling_factor: f32,

        #[arg(long, default_value_t = KmeansOptions::new().get_batch_size())]
        batch_size: u32,

        #[arg(short, long, default_value_t = 0)]
        threads: u8,
    },
    Neuquant {
        #[arg(long, default_value_t = 1)]
        sample_frac: u8,
    },
    Imagequant {
        #[arg(short, long)]
        quality: Option<u8>,

        #[arg(short, long, default_value_t = 0.0)]
        dither_level: f32,

        #[arg(short, long, default_value_t = 0)]
        threads: u8,
    },
    Exoquant {
        #[arg(long)]
        kmeans: bool,

        #[arg(long)]
        dither: bool,
    },
}

#[derive(Parser)]
pub struct Options {
    #[arg(short, long, default_value_t = PaletteSize::MAX, value_parser = parse_palette_size)]
    k: PaletteSize,

    #[arg(long)]
    verbose: bool,

    input: PathBuf,

    #[arg(short, long)]
    output: Option<PathBuf>,

    #[command(subcommand)]
    quantizer: Quantizer,
}

fn parse_palette_size(s: &str) -> Result<PaletteSize, String> {
    let value: u16 = s.parse().map_err(|e| format!("{e}"))?;
    value.try_into().map_err(|e| format!("{e}"))
}

#[allow(clippy::too_many_lines)]
fn main() {
    let Options { quantizer, k, verbose, input, output } = Options::parse();

    macro_rules! log {
        ($name: literal, $val: expr) => {
            if verbose {
                let time = std::time::Instant::now();
                let value = $val;
                println!("{} took {}ms", $name, time.elapsed().as_millis());
                value
            } else {
                $val
            }
        };
    }

    let image = log!("read image", image::open(input).unwrap());

    match quantizer {
        Quantizer::Quantette {
            srgb: false,
            dedup,
            dither,
            dither_error_diffusion,
            kmeans,
            sampling_factor,
            batch_size,
            threads,
        } => {
            let image = Image::try_from(image.into_rgb8()).unwrap();

            let method = if kmeans {
                QuantizeMethod::Kmeans(
                    KmeansOptions::new()
                        .sampling_factor(sampling_factor)
                        .batch_size(batch_size),
                )
            } else {
                QuantizeMethod::Wu
            };

            let parallel = threads != 1;
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(if parallel { threads.into() } else { 0 })
                .build()
                .unwrap();

            let pipeline = Pipeline::new()
                .quantize_method(method)
                .palette_size(k)
                .dedup(dedup)
                .ditherer(
                    dither
                        .then_some(FloydSteinberg::with_error_diffusion(dither_error_diffusion))
                        .flatten(),
                )
                .parallel(parallel);

            if let Some(output) = output {
                let image: RgbImage = log!(
                    "quantization and remapping",
                    pool.install(|| pipeline
                        .input_image(image.as_ref())
                        .output_srgb8_image()
                        .into())
                );
                log!("write image", image.save(output).unwrap())
            } else {
                let colors = log!(
                    "quantization",
                    pool.install(|| pipeline
                        .input_slice(image.as_slice())
                        .unwrap()
                        .output_srgb8_palette())
                );
                print_palette(&colors)
            }
        }
        Quantizer::Quantette {
            srgb: true,
            dedup,
            dither,
            kmeans,
            threads,
            ..
        } => {
            let image = Image::try_from(image.into_rgb8()).unwrap();

            #[allow(clippy::unimplemented)]
            if kmeans || dither || dedup.is_some() {
                unimplemented!("--dither, --kmeans, and --dedup are not compatible with --srgb");
            }

            let parallel = threads != 1;
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(if parallel { threads.into() } else { 0 })
                .build()
                .unwrap();

            if let Some(output) = output {
                let image: RgbImage = log!(
                    "quantization and remapping",
                    pool.install(|| {
                        let binner = BinnerU8x3::rgb();
                        if parallel {
                            let color_map = WuU8x3::run_image_par(image.as_ref(), binner)
                                .unwrap()
                                .parallel_color_map(k);
                            image.map_to_image(color_map)
                        } else {
                            let color_map = WuU8x3::run_image(image.as_ref(), binner)
                                .unwrap()
                                .color_map(k);
                            image.map_to_image(color_map)
                        }
                        .into()
                    })
                );
                log!("write image", image.save(output).unwrap())
            } else {
                let colors = log!(
                    "quantization",
                    pool.install(|| {
                        let binner = BinnerU8x3::rgb();
                        if parallel {
                            WuU8x3::run_image_par(image.as_ref(), binner)
                        } else {
                            WuU8x3::run_image(image.as_ref(), binner)
                        }
                        .unwrap()
                        .palette(k)
                    })
                );
                print_palette(&colors)
            }
        }
        Quantizer::Neuquant { sample_frac } => {
            use color_quant::NeuQuant;

            let image = image.into_rgba8();

            if let Some(output) = output {
                let (nq, indices) = log!("quantization and remapping", {
                    let nq = NeuQuant::new(sample_frac.into(), k.into(), &image);

                    #[allow(clippy::cast_possible_truncation)]
                    let indices = image
                        .chunks_exact(4)
                        .map(|pix| nq.index_of(pix) as u8)
                        .collect::<Vec<_>>();

                    (nq, indices)
                });

                let palette = nq
                    .color_map_rgba()
                    .chunks_exact(4)
                    .map(|c| Srgb::new(c[0], c[1], c[2]))
                    .collect::<Vec<_>>();

                let (width, height) = image.dimensions();
                let image = indexed_parts_to_rgbimage(width, height, palette, indices);
                log!("write image", image.save(output).unwrap())
            } else {
                let nq = log!(
                    "quantization",
                    NeuQuant::new(sample_frac.into(), k.into(), &image)
                );

                let colors = nq
                    .color_map_rgba()
                    .chunks_exact(4)
                    .map(|c| Srgb::new(c[0], c[1], c[2]))
                    .collect::<Vec<_>>();

                print_palette(&colors)
            }
        }
        Quantizer::Imagequant { quality, dither_level, threads } => {
            let image = image.into_rgba8();

            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads.into())
                .build()
                .unwrap();

            pool.install(|| {
                let mut libq = imagequant::new();
                let mut img = libq
                    .new_image(
                        image.as_rgba(),
                        image.width() as usize,
                        image.height() as usize,
                        0.0,
                    )
                    .unwrap();

                if let Some(quality) = quality {
                    libq.set_quality(0, quality).unwrap();
                } else {
                    libq.set_max_colors(k.as_u16().into()).unwrap();
                }

                if let Some(output) = output {
                    let (palette, indices) = log!("quantization and remapping", {
                        let mut quantized = libq.quantize(&mut img).unwrap();
                        quantized.set_dithering_level(dither_level).unwrap();
                        quantized.remapped(&mut img).unwrap()
                    });

                    let palette = palette
                        .into_iter()
                        .map(|c| Srgb::new(c.r, c.g, c.b))
                        .collect::<Vec<_>>();

                    let (width, height) = image.dimensions();
                    let image = indexed_parts_to_rgbimage(width, height, palette, indices);
                    log!("write image", image.save(output).unwrap())
                } else {
                    let mut quantized = log!("quantization", libq.quantize(&mut img).unwrap());

                    let colors = quantized
                        .palette()
                        .iter()
                        .map(|c| Srgb::new(c.r, c.g, c.b))
                        .collect::<Vec<_>>();

                    print_palette(&colors)
                }
            })
        }
        Quantizer::Exoquant { kmeans, dither } => {
            use exoquant::{
                Color, SimpleColorSpace, convert_to_indexed, ditherer, generate_palette, optimizer,
            };

            let image = image.into_rgba8();

            let pixels = image
                .pixels()
                .map(|p| Color::new(p.0[0], p.0[1], p.0[2], p.0[3]))
                .collect::<Vec<_>>();

            let width = image.width() as usize;

            let k = k.into();

            if let Some(output) = output {
                let (palette, indices) = log!(
                    "quantization and remapping",
                    match (kmeans, dither) {
                        (true, true) => convert_to_indexed(
                            &pixels,
                            width,
                            k,
                            &optimizer::KMeans,
                            &ditherer::FloydSteinberg::new(),
                        ),
                        (true, false) => convert_to_indexed(
                            &pixels,
                            width,
                            k,
                            &optimizer::KMeans,
                            &ditherer::None,
                        ),
                        (false, true) => convert_to_indexed(
                            &pixels,
                            width,
                            k,
                            &optimizer::None,
                            &ditherer::FloydSteinberg::new(),
                        ),
                        (false, false) =>
                            convert_to_indexed(&pixels, width, k, &optimizer::None, &ditherer::None),
                    }
                );

                let palette = palette
                    .into_iter()
                    .map(|c| Srgb::new(c.r, c.g, c.b))
                    .collect::<Vec<_>>();

                let (width, height) = image.dimensions();
                let image = indexed_parts_to_rgbimage(width, height, palette, indices);
                log!("write image", image.save(output).unwrap())
            } else {
                let colors = log!(
                    "quantization",
                    if kmeans {
                        generate_palette(
                            &pixels.into_iter().collect(),
                            &SimpleColorSpace::default(),
                            &optimizer::KMeans,
                            k,
                        )
                    } else {
                        generate_palette(
                            &pixels.into_iter().collect(),
                            &SimpleColorSpace::default(),
                            &optimizer::None,
                            k,
                        )
                    }
                );

                let colors = colors
                    .into_iter()
                    .map(|c| Srgb::new(c.r, c.g, c.b))
                    .collect::<Vec<_>>();

                print_palette(&colors)
            }
        }
    }
}

fn indexed_parts_to_rgbimage(
    width: u32,
    height: u32,
    palette: Vec<Srgb<u8>>,
    indices: Vec<u8>,
) -> RgbImage {
    let image = IndexedImage::new(width, height, palette, indices).unwrap();
    image.to_image_par().into()
}

fn print_palette(palette: &[Srgb<u8>]) {
    println!(
        "{}",
        palette
            .iter()
            .map(|color| format!("{color:X}"))
            .collect::<Vec<_>>()
            .join(" ")
    );
}
