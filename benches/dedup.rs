#[path = "../util/mod.rs"]
mod util;

use criterion::{
    Bencher, BenchmarkId, Criterion, SamplingMode, criterion_group, criterion_main,
    measurement::WallTime,
};
use palette::Srgb;
use quantette::{ImageBuf, dedup};
use std::time::Duration;
use util::benchmark_images;

fn bench(
    c: &mut Criterion,
    group: &str,
    mut f: impl FnMut(&mut Bencher<'_, WallTime>, &ImageBuf<Srgb<u8>>),
) {
    let mut group = c.benchmark_group(group);
    group
        .sample_size(30)
        .noise_threshold(0.05)
        .sampling_mode(SamplingMode::Flat)
        .warm_up_time(Duration::from_millis(500));

    for (path, image) in benchmark_images() {
        group.bench_with_input(BenchmarkId::from_parameter(path), image, &mut f);
    }
}

fn dedup_palette_counts_single(c: &mut Criterion) {
    bench(c, "dedup_palette_counts_single", |b, image| {
        b.iter(|| dedup::dedup_colors_u8_3_counts(image.as_slice()))
    });
}

fn dedup_image_counts_single(c: &mut Criterion) {
    bench(c, "dedup_image_counts_single", |b, image| {
        b.iter(|| dedup::dedup_image_u8_3_counts(image.as_ref()))
    });
}

fn dedup_palette_counts_par(c: &mut Criterion) {
    bench(c, "dedup_palette_counts_par", |b, image| {
        b.iter(|| dedup::dedup_colors_u8_3_counts_par(image.as_slice()))
    });
}

fn dedup_image_counts_par(c: &mut Criterion) {
    bench(c, "dedup_image_counts_par", |b, image| {
        b.iter(|| dedup::dedup_image_u8_3_counts_par(image.as_ref()))
    });
}

fn dedup_palette_single(c: &mut Criterion) {
    bench(c, "dedup_palette_single", |b, image| {
        b.iter(|| dedup::dedup_colors_u8_3(image.as_slice()))
    });
}

fn dedup_image_single(c: &mut Criterion) {
    bench(c, "dedup_image_single", |b, image| {
        b.iter(|| dedup::dedup_image_u8_3(image.as_ref()))
    });
}

fn dedup_palette_par(c: &mut Criterion) {
    bench(c, "dedup_palette_par", |b, image| {
        b.iter(|| dedup::dedup_colors_u8_3_par(image.as_slice()))
    });
}

fn dedup_image_par(c: &mut Criterion) {
    bench(c, "dedup_image_par", |b, image| {
        b.iter(|| dedup::dedup_image_u8_3_par(image.as_ref()))
    });
}

criterion_group!(
    benches,
    dedup_palette_counts_single,
    dedup_image_counts_single,
    dedup_palette_counts_par,
    dedup_image_counts_par,
    dedup_palette_single,
    dedup_image_single,
    dedup_palette_par,
    dedup_image_par,
);
criterion_main!(benches);
