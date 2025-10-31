#[path = "../util/mod.rs"]
mod util;

use criterion::{
    Bencher, BenchmarkId, Criterion, SamplingMode, criterion_group, criterion_main,
    measurement::WallTime,
};
use palette::Oklab;
use quantette::{
    IndexedImageCounts, PaletteSize,
    dither::FloydSteinberg,
    wu::{BinnerF32x3, WuF32x3},
};
use std::time::Duration;
use util::benchmark_counts;

fn bench(
    c: &mut Criterion,
    group: &str,
    counts: &[(String, IndexedImageCounts<Oklab, u32>)],
    mut f: impl FnMut(&mut Bencher<'_, WallTime>, &(PaletteSize, &IndexedImageCounts<Oklab, u32>)),
) {
    let mut group = c.benchmark_group(group);
    group
        .sample_size(30)
        .noise_threshold(0.05)
        .sampling_mode(SamplingMode::Flat)
        .warm_up_time(Duration::from_secs(2));

    for k in [
        PaletteSize::MAX,
        PaletteSize::try_from_u8(64).unwrap(),
        PaletteSize::try_from_u8(16).unwrap(),
    ] {
        for (path, counts) in counts {
            group.bench_with_input(BenchmarkId::new(k.to_string(), path), &(k, counts), &mut f);
        }
    }
}

fn dither_indexed_to_indexed(c: &mut Criterion) {
    bench(
        c,
        "dither_indexed_to_indexed",
        benchmark_counts(),
        |b, &(k, image)| {
            let color_map =
                WuF32x3::run_indexed_image_counts_par(image, BinnerF32x3::oklab_from_srgb8())
                    .unwrap()
                    .parallel_color_map(k);

            b.iter(|| FloydSteinberg::new().dither_indexed_to_indexed(image.as_ref(), &color_map))
        },
    )
}

fn dither_indexed_to_indexed_par(c: &mut Criterion) {
    bench(
        c,
        "dither_indexed_to_indexed_par",
        benchmark_counts(),
        |b, &(k, image)| {
            let color_map =
                WuF32x3::run_indexed_image_counts_par(image, BinnerF32x3::oklab_from_srgb8())
                    .unwrap()
                    .parallel_color_map(k);

            b.iter(|| {
                FloydSteinberg::new().dither_indexed_to_indexed_par(image.as_ref(), &color_map)
            })
        },
    )
}

fn dither_indexed_to_image(c: &mut Criterion) {
    bench(
        c,
        "dither_indexed_to_image",
        benchmark_counts(),
        |b, &(k, image)| {
            let color_map =
                WuF32x3::run_indexed_image_counts_par(image, BinnerF32x3::oklab_from_srgb8())
                    .unwrap()
                    .parallel_color_map(k);

            b.iter(|| FloydSteinberg::new().dither_indexed_to_image(image.as_ref(), &color_map))
        },
    )
}

fn dither_indexed_to_image_par(c: &mut Criterion) {
    bench(
        c,
        "dither_indexed_to_image_par",
        benchmark_counts(),
        |b, &(k, image)| {
            let color_map =
                WuF32x3::run_indexed_image_counts_par(image, BinnerF32x3::oklab_from_srgb8())
                    .unwrap()
                    .parallel_color_map(k);

            b.iter(|| FloydSteinberg::new().dither_indexed_to_image_par(image.as_ref(), &color_map))
        },
    )
}

fn dither_image_to_indexed(c: &mut Criterion) {
    bench(
        c,
        "dither_image_to_indexed",
        benchmark_counts(),
        |b, &(k, image)| {
            let color_map =
                WuF32x3::run_indexed_image_counts_par(image, BinnerF32x3::oklab_from_srgb8())
                    .unwrap()
                    .parallel_color_map(k);
            let image = image.to_image_par();

            b.iter(|| FloydSteinberg::new().dither_image_to_indexed(image.as_ref(), &color_map))
        },
    )
}

fn dither_image_to_indexed_par(c: &mut Criterion) {
    bench(
        c,
        "dither_image_to_indexed_par",
        benchmark_counts(),
        |b, &(k, image)| {
            let color_map =
                WuF32x3::run_indexed_image_counts_par(image, BinnerF32x3::oklab_from_srgb8())
                    .unwrap()
                    .parallel_color_map(k);
            let image = image.to_image_par();

            b.iter(|| FloydSteinberg::new().dither_image_to_indexed_par(image.as_ref(), &color_map))
        },
    )
}

fn dither_image_to_image(c: &mut Criterion) {
    bench(
        c,
        "dither_image_to_image",
        benchmark_counts(),
        |b, &(k, image)| {
            let color_map =
                WuF32x3::run_indexed_image_counts_par(image, BinnerF32x3::oklab_from_srgb8())
                    .unwrap()
                    .parallel_color_map(k);
            let image = image.to_image_par();

            b.iter(|| FloydSteinberg::new().dither_image_to_image(image.as_ref(), &color_map))
        },
    )
}

fn dither_image_to_image_par(c: &mut Criterion) {
    bench(
        c,
        "dither_image_to_image_par",
        benchmark_counts(),
        |b, &(k, image)| {
            let color_map =
                WuF32x3::run_indexed_image_counts_par(image, BinnerF32x3::oklab_from_srgb8())
                    .unwrap()
                    .parallel_color_map(k);
            let image = image.to_image_par();

            b.iter(|| FloydSteinberg::new().dither_image_to_image_par(image.as_ref(), &color_map))
        },
    )
}

criterion_group!(
    benches,
    dither_indexed_to_indexed,
    dither_indexed_to_indexed_par,
    dither_indexed_to_image,
    dither_indexed_to_image_par,
    dither_image_to_indexed,
    dither_image_to_indexed_par,
    dither_image_to_image,
    dither_image_to_image_par,
);
criterion_main!(benches);
