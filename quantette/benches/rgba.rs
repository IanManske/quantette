#![allow(clippy::unwrap_used, reason = "benchmark code")]

#[path = "../util/mod.rs"]
mod util;

use criterion::{
    Bencher, BenchmarkId, Criterion, SamplingMode, criterion_group, criterion_main,
    measurement::WallTime,
};
use palette::Srgba;
use quantette::{ImageRef, PaletteSize, RgbaPipeline, RgbaQuantizeMethod, dither::FloydSteinberg};
use rayon::ThreadPoolBuilder;
use std::time::Duration;
use util::alpha_benchmark_images;

fn bench(
    c: &mut Criterion,
    group: &str,
    f: impl Fn(&mut Bencher<'_, WallTime>, PaletteSize, ImageRef<'_, Srgba<u8>>) + Sync,
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
        for (path, image) in alpha_benchmark_images() {
            group.bench_with_input(
                BenchmarkId::new(k.to_string(), path),
                &(k, image),
                |b, (k, image)| pool.install(|| f(b, *k, image.as_ref())),
            );
        }
    }
}

fn rgba_pipeline(
    c: &mut Criterion,
    group: &str,
    method: &RgbaQuantizeMethod,
    ditherer: Option<FloydSteinberg>,
) {
    bench(c, group, |b, k, image| {
        b.iter(|| {
            RgbaPipeline::new()
                .palette_size(k)
                .quantize_method(method.clone())
                .ditherer(ditherer)
                .parallel(true)
                .input_image(image)
                .output_srgba8_indexed_image()
        });
    });
}

fn rgba_wu(c: &mut Criterion) {
    rgba_pipeline(c, "rgba_pipeline_wu", &RgbaQuantizeMethod::Wu, None);
}

fn rgba_kmeans(c: &mut Criterion) {
    rgba_pipeline(
        c,
        "rgba_pipeline_kmeans",
        &RgbaQuantizeMethod::kmeans(),
        None,
    );
}

fn rgba_kmeans_dither(c: &mut Criterion) {
    rgba_pipeline(
        c,
        "rgba_pipeline_kmeans_dither",
        &RgbaQuantizeMethod::kmeans(),
        Some(FloydSteinberg::new()),
    );
}

criterion_group!(benches, rgba_wu, rgba_kmeans, rgba_kmeans_dither);
criterion_main!(benches);
