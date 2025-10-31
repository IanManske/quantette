use clap::{Parser, Subcommand, ValueEnum};
use quantette::{
    ImageBuf, PaletteSize,
    color_space::{oklab_to_srgb8, oklab_to_srgb8_par, srgb8_to_oklab_par},
    dedup,
    kmeans::{Kmeans, KmeansOptions},
    wu::{BinnerF32x3, WuF32x3},
};
use std::{
    cmp::Reverse,
    error::Error,
    fmt,
    fs::File,
    io::{self, BufWriter, Write as _},
    path::PathBuf,
};

#[derive(Clone, Copy, ValueEnum)]
enum Algorithm {
    Wu,
    Online,
    Minibatch,
}

impl fmt::Display for Algorithm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}",
            match self {
                Algorithm::Wu => "wu",
                Algorithm::Online => "online",
                Algorithm::Minibatch => "minibatch",
            }
        )
    }
}

fn parse_palette_size(s: &str) -> Result<PaletteSize, String> {
    let value: u16 = s.parse().map_err(|e| format!("{e}"))?;
    value.try_into().map_err(|e| format!("{e}"))
}

#[derive(Subcommand)]
enum Plot {
    Palette {
        image: PathBuf,

        #[arg(long)]
        pixels_output: Option<PathBuf>,

        #[arg(short = 'n', long, default_value_t = 16)]
        count_threshold: u32,

        #[arg(short = 's', long, default_value_t = 16)]
        sampling_stride: u32,

        #[arg(short, long, default_value_t = PaletteSize::try_from_u8(16).unwrap(), value_parser = parse_palette_size)]
        k: PaletteSize,

        #[arg(long, default_value_t = Algorithm::Wu)]
        algo: Algorithm,

        #[arg(long, default_value_t = KmeansOptions::new().get_sampling_factor())]
        sampling_factor: f32,

        #[arg(long, default_value_t = KmeansOptions::new().get_batch_size())]
        batch_size: u32,
    },
    Freq {
        image: PathBuf,

        #[arg(short, long, default_value_t = 100)]
        limit: u32,
    },
}

#[derive(Parser)]
struct Cli {
    #[command(subcommand)]
    plot: Plot,
}

#[allow(clippy::too_many_lines)]
fn main() -> Result<(), Box<dyn Error>> {
    let Cli { plot } = Cli::parse();

    match plot {
        Plot::Palette {
            image,
            pixels_output,
            count_threshold,
            sampling_stride,
            k,
            algo,
            sampling_factor,
            batch_size,
        } => {
            let image = image::open(image)?.into_rgb8();
            let image = ImageBuf::try_from(image)?;

            let (deduped_image, (palette, counts)) = match algo {
                Algorithm::Wu => {
                    let palette_counts = dedup::dedup_colors_u8_3_counts_par(image.as_slice())?
                        .map(|palette| srgb8_to_oklab_par(&palette));

                    let result = WuF32x3::run_palette_counts_par(
                        &palette_counts,
                        BinnerF32x3::oklab_from_srgb8(),
                    )
                    .unwrap()
                    .palette_and_counts(k);

                    (palette_counts, result)
                }
                Algorithm::Online => {
                    let image = dedup::dedup_image_u8_3_counts_par(image.as_ref())
                        .map(|palette| srgb8_to_oklab_par(&palette));

                    let palette = WuF32x3::run_indexed_image_counts_par(
                        &image,
                        BinnerF32x3::oklab_from_srgb8(),
                    )
                    .unwrap()
                    .palette(k);

                    let options = KmeansOptions::new()
                        .sampling_factor(sampling_factor)
                        .batch_size(batch_size);
                    let result = Kmeans::run_indexed_image(image.as_ref(), palette, options)
                        .into_palette_and_counts();

                    (image.into_palette_counts(), result)
                }
                Algorithm::Minibatch => {
                    let image = dedup::dedup_image_u8_3_counts_par(image.as_ref())
                        .map(|palette| srgb8_to_oklab_par(&palette));

                    let palette = WuF32x3::run_indexed_image_counts_par(
                        &image,
                        BinnerF32x3::oklab_from_srgb8(),
                    )
                    .unwrap()
                    .palette(k);

                    let options = KmeansOptions::new()
                        .sampling_factor(sampling_factor)
                        .batch_size(batch_size);
                    let result = Kmeans::run_indexed_image_par(image.as_ref(), palette, options)
                        .into_palette_and_counts();

                    (image.into_palette_counts(), result)
                }
            };

            if let Some(pixel_output) = pixels_output {
                let deduped_srgb = oklab_to_srgb8_par(deduped_image.palette());
                let mut out = File::create(pixel_output)?;

                for ((oklab, &n), srgb) in deduped_image
                    .palette()
                    .iter()
                    .zip(deduped_image.counts())
                    .zip(deduped_srgb)
                    .step_by(sampling_stride as usize)
                {
                    if n >= count_threshold {
                        let (l, a, b) = oklab.into_components();
                        out.write_all(&a.to_ne_bytes())?;
                        out.write_all(&b.to_ne_bytes())?;
                        out.write_all(&l.to_ne_bytes())?;
                        out.write_all(&u32::from(srgb).to_ne_bytes())?;
                    }
                }
            }

            let palette_srgb = oklab_to_srgb8(&palette);
            let mut out = BufWriter::new(io::stdout());

            writeln!(out, "a b l n color")?;
            for ((centroid, n), srgb) in palette.into_iter().zip(counts).zip(palette_srgb) {
                let (l, a, b) = centroid.into_components();
                writeln!(out, "{a} {b} {l} {n} 0x{srgb:X}")?;
            }
        }
        Plot::Freq { image, limit } => {
            let image = image::open(image)?.into_rgb8();
            let image = ImageBuf::try_from(image)?;
            let (palette, counts) = dedup::dedup_image_u8_3_counts_par(image.as_ref())
                .into_palette_counts()
                .into_parts();

            let mut sorted = palette.into_iter().zip(counts).collect::<Vec<_>>();
            sorted.sort_by_key(|&(_, n)| Reverse(n));
            sorted.truncate(limit as usize);

            let mut out = BufWriter::new(io::stdout());
            for (color, n) in sorted {
                out.write_all(&u32::from(color).to_ne_bytes())?;
                out.write_all(&n.to_ne_bytes())?;
            }
        }
    }

    Ok(())
}
