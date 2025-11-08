#[path = "../util/mod.rs"]
mod util;

use criterion::{
    Bencher, BenchmarkId, Criterion, SamplingMode, criterion_group, criterion_main,
    measurement::WallTime,
};
use palette::Srgb;
use quantette::{
    ImageBuf, ImageRef, PaletteSize, Pipeline, QuantizeMethod,
    color_space::srgb8_to_oklab_par,
    dither::FloydSteinberg,
    kmeans::KmeansOptions,
    wu::{BinnerF32x3, WuF32x3},
};
use rayon::ThreadPoolBuilder;
use std::time::Duration;
use util::benchmark_images;

fn bench(
    c: &mut Criterion,
    group: &str,
    counts: &[(String, ImageBuf<Srgb<u8>>)],
    f: impl Fn(&mut Bencher<'_, WallTime>, PaletteSize, ImageRef<'_, Srgb<u8>>) + Sync,
) {
    let mut group = c.benchmark_group(group);
    group
        .sample_size(30)
        .noise_threshold(0.05)
        .sampling_mode(SamplingMode::Flat)
        .warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_secs(1));

    let pool = ThreadPoolBuilder::new().num_threads(0).build().unwrap();

    for k in [
        PaletteSize::MAX,
        PaletteSize::try_from_u8(64).unwrap(),
        PaletteSize::try_from_u8(16).unwrap(),
    ] {
        for (path, counts) in counts {
            group.bench_with_input(
                BenchmarkId::new(k.to_string(), path),
                &(k, counts),
                |b, (k, image)| pool.install(|| f(b, *k, image.as_ref())),
            );
        }
    }
}

fn palette_wu_single(c: &mut Criterion) {
    bench(
        c,
        "pipeline_palette_wu_single",
        benchmark_images(),
        |b, k, image| {
            b.iter(|| {
                Pipeline::new()
                    .palette_size(k)
                    .input_image(image.as_ref())
                    .output_oklab_palette()
            })
        },
    )
}

fn palette_wu_par(c: &mut Criterion) {
    bench(
        c,
        "pipeline_palette_wu_par",
        benchmark_images(),
        |b, k, image| {
            b.iter(|| {
                Pipeline::new()
                    .palette_size(k)
                    .parallel(true)
                    .input_image(image.as_ref())
                    .output_oklab_palette()
            })
        },
    )
}

fn palette_kmeans_single(c: &mut Criterion) {
    bench(
        c,
        "pipeline_palette_kmeans_single",
        benchmark_images(),
        |b, k, image| {
            b.iter(|| {
                Pipeline::new()
                    .palette_size(k)
                    .quantize_method(QuantizeMethod::kmeans())
                    .input_image(image.as_ref())
                    .output_oklab_palette()
            })
        },
    )
}

fn palette_kmeans_par(c: &mut Criterion) {
    bench(
        c,
        "pipeline_palette_kmeans_par",
        benchmark_images(),
        |b, k, image| {
            b.iter(|| {
                Pipeline::new()
                    .palette_size(k)
                    .quantize_method(QuantizeMethod::kmeans())
                    .parallel(true)
                    .input_image(image.as_ref())
                    .output_oklab_palette()
            })
        },
    )
}

fn srgb_image_custom_single(c: &mut Criterion) {
    bench(
        c,
        "pipeline_srgb_image_custom_single",
        benchmark_images(),
        |b, k, image| {
            let oklab = image.map_ref(srgb8_to_oklab_par);
            let palette = WuF32x3::run_image_par(oklab.as_ref(), BinnerF32x3::oklab_from_srgb8())
                .unwrap()
                .palette(k);
            b.iter(|| {
                Pipeline::new()
                    .ditherer(None)
                    .palette_size(k)
                    .quantize_method(palette.clone())
                    .input_image(image.as_ref())
                    .output_srgb8_image()
            })
        },
    )
}

fn srgb_image_custom_par(c: &mut Criterion) {
    bench(
        c,
        "pipeline_srgb_image_custom_par",
        benchmark_images(),
        |b, k, image| {
            let oklab = image.map_ref(srgb8_to_oklab_par);
            let palette = WuF32x3::run_image_par(oklab.as_ref(), BinnerF32x3::oklab_from_srgb8())
                .unwrap()
                .palette(k);
            b.iter(|| {
                Pipeline::new()
                    .ditherer(None)
                    .palette_size(k)
                    .quantize_method(palette.clone())
                    .parallel(true)
                    .input_image(image.as_ref())
                    .output_srgb8_image()
            })
        },
    )
}

fn srgb_image_custom_dither_single(c: &mut Criterion) {
    bench(
        c,
        "pipeline_srgb_image_custom_dither_single",
        benchmark_images(),
        |b, k, image| {
            let oklab = image.map_ref(srgb8_to_oklab_par);
            let palette = WuF32x3::run_image_par(oklab.as_ref(), BinnerF32x3::oklab_from_srgb8())
                .unwrap()
                .palette(k);
            b.iter(|| {
                Pipeline::new()
                    .ditherer(FloydSteinberg::new())
                    .palette_size(k)
                    .quantize_method(palette.clone())
                    .input_image(image.as_ref())
                    .output_srgb8_image()
            })
        },
    )
}

fn srgb_image_custom_dither_par(c: &mut Criterion) {
    bench(
        c,
        "pipeline_srgb_image_custom_dither_par",
        benchmark_images(),
        |b, k, image| {
            let oklab = image.map_ref(srgb8_to_oklab_par);
            let palette = WuF32x3::run_image_par(oklab.as_ref(), BinnerF32x3::oklab_from_srgb8())
                .unwrap()
                .palette(k);
            b.iter(|| {
                Pipeline::new()
                    .ditherer(FloydSteinberg::new())
                    .palette_size(k)
                    .quantize_method(palette.clone())
                    .parallel(true)
                    .input_image(image.as_ref())
                    .output_srgb8_image()
            })
        },
    )
}

fn srgb_image_wu_single(c: &mut Criterion) {
    bench(
        c,
        "pipeline_srgb_image_wu_single",
        benchmark_images(),
        |b, k, image| {
            b.iter(|| {
                Pipeline::new()
                    .ditherer(None)
                    .palette_size(k)
                    .input_image(image.as_ref())
                    .output_srgb8_image()
            })
        },
    )
}

fn srgb_image_wu_par(c: &mut Criterion) {
    bench(
        c,
        "pipeline_srgb_image_wu_par",
        benchmark_images(),
        |b, k, image| {
            b.iter(|| {
                Pipeline::new()
                    .ditherer(None)
                    .palette_size(k)
                    .parallel(true)
                    .input_image(image.as_ref())
                    .output_srgb8_image()
            })
        },
    )
}

fn srgb_image_wu_dither_single(c: &mut Criterion) {
    bench(
        c,
        "pipeline_srgb_image_wu_dither_single",
        benchmark_images(),
        |b, k, image| {
            b.iter(|| {
                Pipeline::new()
                    .ditherer(FloydSteinberg::new())
                    .palette_size(k)
                    .input_image(image.as_ref())
                    .output_srgb8_image()
            })
        },
    )
}

fn srgb_image_wu_dither_par(c: &mut Criterion) {
    bench(
        c,
        "pipeline_srgb_image_wu_dither_par",
        benchmark_images(),
        |b, k, image| {
            b.iter(|| {
                Pipeline::new()
                    .ditherer(FloydSteinberg::new())
                    .palette_size(k)
                    .parallel(true)
                    .input_image(image.as_ref())
                    .output_srgb8_image()
            })
        },
    )
}

fn srgb_image_kmeans_single(c: &mut Criterion) {
    bench(
        c,
        "pipeline_srgb_image_kmeans_single",
        benchmark_images(),
        |b, k, image| {
            b.iter(|| {
                Pipeline::new()
                    .ditherer(None)
                    .palette_size(k)
                    .quantize_method(KmeansOptions::new())
                    .input_image(image.as_ref())
                    .output_srgb8_image()
            })
        },
    )
}

fn srgb_image_kmeans_par(c: &mut Criterion) {
    bench(
        c,
        "pipeline_srgb_image_kmeans_par",
        benchmark_images(),
        |b, k, image| {
            b.iter(|| {
                Pipeline::new()
                    .ditherer(None)
                    .palette_size(k)
                    .quantize_method(KmeansOptions::new())
                    .parallel(true)
                    .input_image(image.as_ref())
                    .output_srgb8_image()
            })
        },
    )
}

fn srgb_image_kmeans_dither_single(c: &mut Criterion) {
    bench(
        c,
        "pipeline_srgb_image_kmeans_dither_single",
        benchmark_images(),
        |b, k, image| {
            b.iter(|| {
                Pipeline::new()
                    .ditherer(FloydSteinberg::new())
                    .palette_size(k)
                    .quantize_method(KmeansOptions::new())
                    .input_image(image.as_ref())
                    .output_srgb8_image()
            })
        },
    )
}

fn srgb_image_kmeans_dither_par(c: &mut Criterion) {
    bench(
        c,
        "pipeline_srgb_image_kmeans_dither_par",
        benchmark_images(),
        |b, k, image| {
            b.iter(|| {
                Pipeline::new()
                    .ditherer(FloydSteinberg::new())
                    .palette_size(k)
                    .quantize_method(KmeansOptions::new())
                    .parallel(true)
                    .input_image(image.as_ref())
                    .output_srgb8_image()
            })
        },
    )
}

fn oklab_image_custom_single(c: &mut Criterion) {
    bench(
        c,
        "pipeline_oklab_image_custom_single",
        benchmark_images(),
        |b, k, image| {
            let oklab = image.map_ref(srgb8_to_oklab_par);
            let palette = WuF32x3::run_image_par(oklab.as_ref(), BinnerF32x3::oklab_from_srgb8())
                .unwrap()
                .palette(k);
            b.iter(|| {
                Pipeline::new()
                    .ditherer(None)
                    .palette_size(k)
                    .quantize_method(palette.clone())
                    .input_image(image.as_ref())
                    .output_oklab_image()
            })
        },
    )
}

fn oklab_image_custom_par(c: &mut Criterion) {
    bench(
        c,
        "pipeline_oklab_image_custom_par",
        benchmark_images(),
        |b, k, image| {
            let oklab = image.map_ref(srgb8_to_oklab_par);
            let palette = WuF32x3::run_image_par(oklab.as_ref(), BinnerF32x3::oklab_from_srgb8())
                .unwrap()
                .palette(k);
            b.iter(|| {
                Pipeline::new()
                    .ditherer(None)
                    .palette_size(k)
                    .quantize_method(palette.clone())
                    .parallel(true)
                    .input_image(image.as_ref())
                    .output_oklab_image()
            })
        },
    )
}

fn oklab_image_custom_dither_single(c: &mut Criterion) {
    bench(
        c,
        "pipeline_oklab_image_custom_dither_single",
        benchmark_images(),
        |b, k, image| {
            let oklab = image.map_ref(srgb8_to_oklab_par);
            let palette = WuF32x3::run_image_par(oklab.as_ref(), BinnerF32x3::oklab_from_srgb8())
                .unwrap()
                .palette(k);
            b.iter(|| {
                Pipeline::new()
                    .ditherer(FloydSteinberg::new())
                    .palette_size(k)
                    .quantize_method(palette.clone())
                    .input_image(image.as_ref())
                    .output_oklab_image()
            })
        },
    )
}

fn oklab_image_custom_dither_par(c: &mut Criterion) {
    bench(
        c,
        "pipeline_oklab_image_custom_dither_par",
        benchmark_images(),
        |b, k, image| {
            let oklab = image.map_ref(srgb8_to_oklab_par);
            let palette = WuF32x3::run_image_par(oklab.as_ref(), BinnerF32x3::oklab_from_srgb8())
                .unwrap()
                .palette(k);
            b.iter(|| {
                Pipeline::new()
                    .ditherer(FloydSteinberg::new())
                    .palette_size(k)
                    .quantize_method(palette.clone())
                    .parallel(true)
                    .input_image(image.as_ref())
                    .output_oklab_image()
            })
        },
    )
}

fn oklab_image_wu_single(c: &mut Criterion) {
    bench(
        c,
        "pipeline_oklab_image_wu_single",
        benchmark_images(),
        |b, k, image| {
            b.iter(|| {
                Pipeline::new()
                    .ditherer(None)
                    .palette_size(k)
                    .input_image(image.as_ref())
                    .output_oklab_image()
            })
        },
    )
}

fn oklab_image_wu_par(c: &mut Criterion) {
    bench(
        c,
        "pipeline_oklab_image_wu_par",
        benchmark_images(),
        |b, k, image| {
            b.iter(|| {
                Pipeline::new()
                    .ditherer(None)
                    .palette_size(k)
                    .parallel(true)
                    .input_image(image.as_ref())
                    .output_oklab_image()
            })
        },
    )
}

fn oklab_image_wu_dither_single(c: &mut Criterion) {
    bench(
        c,
        "pipeline_oklab_image_wu_dither_single",
        benchmark_images(),
        |b, k, image| {
            b.iter(|| {
                Pipeline::new()
                    .ditherer(FloydSteinberg::new())
                    .palette_size(k)
                    .input_image(image.as_ref())
                    .output_oklab_image()
            })
        },
    )
}

fn oklab_image_wu_dither_par(c: &mut Criterion) {
    bench(
        c,
        "pipeline_oklab_image_wu_dither_par",
        benchmark_images(),
        |b, k, image| {
            b.iter(|| {
                Pipeline::new()
                    .ditherer(FloydSteinberg::new())
                    .palette_size(k)
                    .parallel(true)
                    .input_image(image.as_ref())
                    .output_oklab_image()
            })
        },
    )
}

fn oklab_image_kmeans_single(c: &mut Criterion) {
    bench(
        c,
        "pipeline_oklab_image_kmeans_single",
        benchmark_images(),
        |b, k, image| {
            b.iter(|| {
                Pipeline::new()
                    .ditherer(None)
                    .palette_size(k)
                    .quantize_method(KmeansOptions::new())
                    .input_image(image.as_ref())
                    .output_oklab_image()
            })
        },
    )
}

fn oklab_image_kmeans_par(c: &mut Criterion) {
    bench(
        c,
        "pipeline_oklab_image_kmeans_par",
        benchmark_images(),
        |b, k, image| {
            b.iter(|| {
                Pipeline::new()
                    .ditherer(None)
                    .palette_size(k)
                    .quantize_method(KmeansOptions::new())
                    .parallel(true)
                    .input_image(image.as_ref())
                    .output_oklab_image()
            })
        },
    )
}

fn oklab_image_kmeans_dither_single(c: &mut Criterion) {
    bench(
        c,
        "pipeline_oklab_image_kmeans_dither_single",
        benchmark_images(),
        |b, k, image| {
            b.iter(|| {
                Pipeline::new()
                    .ditherer(FloydSteinberg::new())
                    .quantize_method(KmeansOptions::new())
                    .palette_size(k)
                    .input_image(image.as_ref())
                    .output_oklab_image()
            })
        },
    )
}

fn oklab_image_kmeans_dither_par(c: &mut Criterion) {
    bench(
        c,
        "pipeline_oklab_image_kmeans_dither_par",
        benchmark_images(),
        |b, k, image| {
            b.iter(|| {
                Pipeline::new()
                    .ditherer(FloydSteinberg::new())
                    .palette_size(k)
                    .quantize_method(KmeansOptions::new())
                    .parallel(true)
                    .input_image(image.as_ref())
                    .output_oklab_image()
            })
        },
    )
}

fn indexed_image_wu_single(c: &mut Criterion) {
    bench(
        c,
        "pipeline_indexed_image_wu_single",
        benchmark_images(),
        |b, k, image| {
            b.iter(|| {
                Pipeline::new()
                    .ditherer(None)
                    .palette_size(k)
                    .input_image(image.as_ref())
                    .output_oklab_indexed_image()
            })
        },
    )
}

fn indexed_image_wu_par(c: &mut Criterion) {
    bench(
        c,
        "pipeline_indexed_image_wu_par",
        benchmark_images(),
        |b, k, image| {
            b.iter(|| {
                Pipeline::new()
                    .ditherer(None)
                    .palette_size(k)
                    .parallel(true)
                    .input_image(image.as_ref())
                    .output_oklab_indexed_image()
            })
        },
    )
}

fn indexed_image_wu_dither_single(c: &mut Criterion) {
    bench(
        c,
        "pipeline_indexed_image_wu_dither_single",
        benchmark_images(),
        |b, k, image| {
            b.iter(|| {
                Pipeline::new()
                    .ditherer(FloydSteinberg::new())
                    .palette_size(k)
                    .input_image(image.as_ref())
                    .output_oklab_indexed_image()
            })
        },
    )
}

fn indexed_image_wu_dither_par(c: &mut Criterion) {
    bench(
        c,
        "pipeline_indexed_image_wu_dither_par",
        benchmark_images(),
        |b, k, image| {
            b.iter(|| {
                Pipeline::new()
                    .ditherer(FloydSteinberg::new())
                    .palette_size(k)
                    .parallel(true)
                    .input_image(image.as_ref())
                    .output_oklab_indexed_image()
            })
        },
    )
}

fn indexed_image_custom_single(c: &mut Criterion) {
    bench(
        c,
        "pipeline_indexed_image_custom_single",
        benchmark_images(),
        |b, k, image| {
            let oklab = image.map_ref(srgb8_to_oklab_par);
            let palette = WuF32x3::run_image_par(oklab.as_ref(), BinnerF32x3::oklab_from_srgb8())
                .unwrap()
                .palette(k);
            b.iter(|| {
                Pipeline::new()
                    .ditherer(None)
                    .palette_size(k)
                    .quantize_method(palette.clone())
                    .input_image(image.as_ref())
                    .output_oklab_indexed_image()
            })
        },
    )
}

fn indexed_image_custom_par(c: &mut Criterion) {
    bench(
        c,
        "pipeline_indexed_image_custom_par",
        benchmark_images(),
        |b, k, image| {
            let oklab = image.map_ref(srgb8_to_oklab_par);
            let palette = WuF32x3::run_image_par(oklab.as_ref(), BinnerF32x3::oklab_from_srgb8())
                .unwrap()
                .palette(k);
            b.iter(|| {
                Pipeline::new()
                    .ditherer(None)
                    .palette_size(k)
                    .parallel(true)
                    .quantize_method(palette.clone())
                    .input_image(image.as_ref())
                    .output_oklab_indexed_image()
            })
        },
    )
}

fn indexed_image_custom_dither_single(c: &mut Criterion) {
    bench(
        c,
        "pipeline_indexed_image_custom_dither_single",
        benchmark_images(),
        |b, k, image| {
            let oklab = image.map_ref(srgb8_to_oklab_par);
            let palette = WuF32x3::run_image_par(oklab.as_ref(), BinnerF32x3::oklab_from_srgb8())
                .unwrap()
                .palette(k);
            b.iter(|| {
                Pipeline::new()
                    .ditherer(FloydSteinberg::new())
                    .palette_size(k)
                    .quantize_method(palette.clone())
                    .input_image(image.as_ref())
                    .output_oklab_indexed_image()
            })
        },
    )
}

fn indexed_image_custom_dither_par(c: &mut Criterion) {
    bench(
        c,
        "pipeline_indexed_image_custom_dither_par",
        benchmark_images(),
        |b, k, image| {
            let oklab = image.map_ref(srgb8_to_oklab_par);
            let palette = WuF32x3::run_image_par(oklab.as_ref(), BinnerF32x3::oklab_from_srgb8())
                .unwrap()
                .palette(k);
            b.iter(|| {
                Pipeline::new()
                    .ditherer(FloydSteinberg::new())
                    .palette_size(k)
                    .quantize_method(palette.clone())
                    .parallel(true)
                    .input_image(image.as_ref())
                    .output_oklab_indexed_image()
            })
        },
    )
}

fn indexed_image_kmeans_single(c: &mut Criterion) {
    bench(
        c,
        "pipeline_indexed_image_kmeans_single",
        benchmark_images(),
        |b, k, image| {
            b.iter(|| {
                Pipeline::new()
                    .ditherer(None)
                    .palette_size(k)
                    .quantize_method(KmeansOptions::new())
                    .input_image(image.as_ref())
                    .output_oklab_indexed_image()
            })
        },
    )
}

fn indexed_image_kmeans_par(c: &mut Criterion) {
    bench(
        c,
        "pipeline_indexed_image_kmeans_par",
        benchmark_images(),
        |b, k, image| {
            b.iter(|| {
                Pipeline::new()
                    .ditherer(None)
                    .palette_size(k)
                    .quantize_method(KmeansOptions::new())
                    .parallel(true)
                    .input_image(image.as_ref())
                    .output_oklab_indexed_image()
            })
        },
    )
}

fn indexed_image_kmeans_dither_single(c: &mut Criterion) {
    bench(
        c,
        "pipeline_indexed_image_kmeans_dither_single",
        benchmark_images(),
        |b, k, image| {
            b.iter(|| {
                Pipeline::new()
                    .ditherer(FloydSteinberg::new())
                    .palette_size(k)
                    .quantize_method(KmeansOptions::new())
                    .input_image(image.as_ref())
                    .output_oklab_indexed_image()
            })
        },
    )
}

fn indexed_image_kmeans_dither_par(c: &mut Criterion) {
    bench(
        c,
        "pipeline_indexed_image_kmeans_dither_par",
        benchmark_images(),
        |b, k, image| {
            b.iter(|| {
                Pipeline::new()
                    .ditherer(FloydSteinberg::new())
                    .palette_size(k)
                    .parallel(true)
                    .quantize_method(KmeansOptions::new())
                    .input_image(image.as_ref())
                    .output_oklab_indexed_image()
            })
        },
    )
}

criterion_group!(
    benches,
    palette_wu_single,
    palette_wu_par,
    palette_kmeans_single,
    palette_kmeans_par,
    srgb_image_custom_single,
    srgb_image_custom_par,
    srgb_image_custom_dither_single,
    srgb_image_custom_dither_par,
    srgb_image_wu_single,
    srgb_image_wu_par,
    srgb_image_wu_dither_single,
    srgb_image_wu_dither_par,
    srgb_image_kmeans_single,
    srgb_image_kmeans_par,
    srgb_image_kmeans_dither_single,
    srgb_image_kmeans_dither_par,
    oklab_image_custom_single,
    oklab_image_custom_par,
    oklab_image_custom_dither_single,
    oklab_image_custom_dither_par,
    oklab_image_wu_single,
    oklab_image_wu_par,
    oklab_image_wu_dither_single,
    oklab_image_wu_dither_par,
    oklab_image_kmeans_single,
    oklab_image_kmeans_par,
    oklab_image_kmeans_dither_single,
    oklab_image_kmeans_dither_par,
    indexed_image_custom_single,
    indexed_image_custom_par,
    indexed_image_custom_dither_single,
    indexed_image_custom_dither_par,
    indexed_image_wu_single,
    indexed_image_wu_par,
    indexed_image_wu_dither_single,
    indexed_image_wu_dither_par,
    indexed_image_kmeans_single,
    indexed_image_kmeans_par,
    indexed_image_kmeans_dither_single,
    indexed_image_kmeans_dither_par,
);
criterion_main!(benches);
