#[path = "../util/mod.rs"]
mod util;

use criterion::{
    Bencher, BenchmarkId, Criterion, SamplingMode, criterion_group, criterion_main,
    measurement::WallTime,
};
use quantette::{
    PaletteSize,
    color_space::srgb8_to_oklab_par,
    kmeans::{Kmeans, KmeansOptions},
    wu::{BinnerF32x3, WuF32x3},
};
use std::time::Duration;
use util::{benchmark_counts, benchmark_images};

fn bench<T>(
    c: &mut Criterion,
    group: &str,
    counts: &[(String, T)],
    mut f: impl FnMut(&mut Bencher<'_, WallTime>, &(PaletteSize, &T)),
) {
    let mut group = c.benchmark_group(group);
    group
        .sample_size(30)
        .noise_threshold(0.05)
        .sampling_mode(SamplingMode::Flat)
        .warm_up_time(Duration::from_millis(500));

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

fn kmeans_palette_single(c: &mut Criterion) {
    bench(
        c,
        "kmeans_palette_single",
        benchmark_images(),
        |b, &(k, image)| {
            let image = image.map_ref(srgb8_to_oklab_par);
            let centroids = WuF32x3::run_image_par(image.as_ref(), BinnerF32x3::oklab_from_srgb8())
                .unwrap()
                .palette(k);

            b.iter(|| {
                Kmeans::run_image(image.as_ref(), centroids.clone(), KmeansOptions::new())
                    .into_palette()
            })
        },
    )
}

fn kmeans_dedup_palette_single(c: &mut Criterion) {
    bench(
        c,
        "kmeans_dedup_palette_single",
        benchmark_counts(),
        |b, &(k, counts)| {
            let centroids =
                WuF32x3::run_indexed_image_counts_par(counts, BinnerF32x3::oklab_from_srgb8())
                    .unwrap()
                    .palette(k);

            b.iter(|| {
                Kmeans::run_indexed_image(counts.as_ref(), centroids.clone(), KmeansOptions::new())
                    .into_palette()
            })
        },
    )
}

fn kmeans_remap_single(c: &mut Criterion) {
    bench(
        c,
        "kmeans_remap_single",
        benchmark_images(),
        |b, &(k, image)| {
            let image = image.map_ref(srgb8_to_oklab_par);
            let centroids = WuF32x3::run_image_par(image.as_ref(), BinnerF32x3::oklab_from_srgb8())
                .unwrap()
                .palette(k);

            b.iter(|| {
                let color_map =
                    Kmeans::run_image(image.as_ref(), centroids.clone(), KmeansOptions::new())
                        .into_color_map();
                image.map_to_image(color_map)
            })
        },
    )
}

fn kmeans_dedup_remap_single(c: &mut Criterion) {
    bench(
        c,
        "kmeans_dedup_remap_single",
        benchmark_counts(),
        |b, &(k, counts)| {
            let centroids =
                WuF32x3::run_indexed_image_counts_par(counts, BinnerF32x3::oklab_from_srgb8())
                    .unwrap()
                    .palette(k);

            b.iter(|| {
                let color_map = Kmeans::run_indexed_image(
                    counts.as_ref(),
                    centroids.clone(),
                    KmeansOptions::new(),
                )
                .into_color_map();
                counts.map_to_image(color_map)
            })
        },
    )
}

fn kmeans_palette_par(c: &mut Criterion) {
    bench(
        c,
        "kmeans_palette_par",
        benchmark_images(),
        |b, &(k, image)| {
            let image = image.map_ref(srgb8_to_oklab_par);
            let centroids = WuF32x3::run_image_par(image.as_ref(), BinnerF32x3::oklab_from_srgb8())
                .unwrap()
                .palette(k);

            b.iter(|| {
                Kmeans::run_image_par(image.as_ref(), centroids.clone(), KmeansOptions::new())
                    .into_palette()
            })
        },
    )
}

fn kmeans_dedup_palette_par(c: &mut Criterion) {
    bench(
        c,
        "kmeans_dedup_palette_par",
        benchmark_counts(),
        |b, &(k, counts)| {
            let centroids =
                WuF32x3::run_indexed_image_counts_par(counts, BinnerF32x3::oklab_from_srgb8())
                    .unwrap()
                    .palette(k);

            b.iter(|| {
                Kmeans::run_indexed_image_par(
                    counts.as_ref(),
                    centroids.clone(),
                    KmeansOptions::new(),
                )
                .into_palette()
            })
        },
    )
}

fn kmeans_remap_par(c: &mut Criterion) {
    bench(
        c,
        "kmeans_remap_par",
        benchmark_images(),
        |b, &(k, image)| {
            let image = image.map_ref(srgb8_to_oklab_par);
            let centroids = WuF32x3::run_image_par(image.as_ref(), BinnerF32x3::oklab_from_srgb8())
                .unwrap()
                .palette(k);

            b.iter(|| {
                let color_map =
                    Kmeans::run_image_par(image.as_ref(), centroids.clone(), KmeansOptions::new())
                        .into_parallel_color_map();
                image.map_to_image(color_map)
            })
        },
    )
}

fn kmeans_dedup_remap_par(c: &mut Criterion) {
    bench(
        c,
        "kmeans_dedup_remap_par",
        benchmark_counts(),
        |b, &(k, counts)| {
            let centroids =
                WuF32x3::run_indexed_image_counts_par(counts, BinnerF32x3::oklab_from_srgb8())
                    .unwrap()
                    .palette(k);

            b.iter(|| {
                let color_map = Kmeans::run_indexed_image_par(
                    counts.as_ref(),
                    centroids.clone(),
                    KmeansOptions::new(),
                )
                .into_parallel_color_map();
                counts.map_to_image_par(color_map)
            })
        },
    )
}

criterion_group!(
    benches,
    kmeans_palette_single,
    kmeans_dedup_palette_single,
    kmeans_remap_single,
    kmeans_dedup_remap_single,
    kmeans_palette_par,
    kmeans_dedup_palette_par,
    kmeans_remap_par,
    kmeans_dedup_remap_par,
);
criterion_main!(benches);
