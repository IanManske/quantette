#[path = "../util/mod.rs"]
mod util;

use criterion::{
    Bencher, BenchmarkId, Criterion, SamplingMode, criterion_group, criterion_main,
    measurement::WallTime,
};
use palette::Oklab;
use quantette::{
    IndexedImageCounts, PaletteSize,
    color_map::{NearestNeighborColorMap, NearestNeighborParallelColorMap},
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

fn remap_single(c: &mut Criterion) {
    bench(
        c,
        "nearest_neighbor_single",
        benchmark_counts(),
        |b, &(k, counts)| {
            let palette =
                WuF32x3::run_indexed_image_counts_par(counts, BinnerF32x3::oklab_from_srgb8())
                    .unwrap()
                    .palette(k);
            let color_map = NearestNeighborColorMap::new(palette);
            b.iter(|| counts.map_to_indexed(&color_map))
        },
    )
}

fn remap_par(c: &mut Criterion) {
    bench(
        c,
        "nearest_neighbor_par",
        benchmark_counts(),
        |b, &(k, counts)| {
            let palette =
                WuF32x3::run_indexed_image_counts_par(counts, BinnerF32x3::oklab_from_srgb8())
                    .unwrap()
                    .palette(k);
            let color_map = NearestNeighborParallelColorMap::new(palette);
            b.iter(|| counts.map_to_indexed_par(&color_map))
        },
    )
}

criterion_group!(benches, remap_single, remap_par);
criterion_main!(benches);
