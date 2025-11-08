#[path = "../util/mod.rs"]
mod util;

use criterion::{
    Bencher, BenchmarkId, Criterion, SamplingMode, criterion_group, criterion_main,
    measurement::WallTime,
};
use quantette::{
    PaletteSize,
    color_space::srgb8_to_oklab_par,
    dedup,
    wu::{BinnerF32x3, BinnerU8x3, WuF32x3, WuU8x3},
};
use std::time::Duration;
use util::{benchmark_counts, benchmark_images};

// Wu running time scales with image size and is barely affected by palette size.
const K: PaletteSize = PaletteSize::MAX;

fn bench<T>(
    c: &mut Criterion,
    group: &str,
    counts: &[(String, T)],
    mut f: impl FnMut(&mut Bencher<'_, WallTime>, &T),
) {
    let mut group = c.benchmark_group(group);
    group
        .sample_size(30)
        .noise_threshold(0.05)
        .sampling_mode(SamplingMode::Flat)
        .warm_up_time(Duration::from_millis(500));

    for (path, counts) in counts {
        group.bench_with_input(BenchmarkId::from_parameter(path), counts, &mut f);
    }
}

fn wu_srgb_palette_single(c: &mut Criterion) {
    bench(
        c,
        "wu_srgb_palette_single",
        benchmark_images(),
        |b, image| {
            b.iter(|| {
                WuU8x3::run_image(image.as_ref(), BinnerU8x3::rgb())
                    .unwrap()
                    .palette(K)
            })
        },
    )
}

fn wu_srgb_dedup_palette_single(c: &mut Criterion) {
    bench(
        c,
        "wu_srgb_dedup_palette_single",
        benchmark_images(),
        |b, image| {
            let image = dedup::dedup_image_u8_3_counts_par(image.as_ref());
            b.iter(|| {
                WuU8x3::run_indexed_image_counts(&image, BinnerU8x3::rgb())
                    .unwrap()
                    .palette(K)
            })
        },
    )
}

fn wu_srgb_remap_single(c: &mut Criterion) {
    bench(c, "wu_srgb_remap_single", benchmark_images(), |b, image| {
        b.iter(|| {
            let color_map = WuU8x3::run_image(image.as_ref(), BinnerU8x3::rgb())
                .unwrap()
                .color_map(K);

            image.map_to_image(color_map)
        })
    })
}

fn wu_srgb_dedup_remap_single(c: &mut Criterion) {
    bench(
        c,
        "wu_srgb_dedup_remap_single",
        benchmark_images(),
        |b, image| {
            let image = dedup::dedup_image_u8_3_counts_par(image.as_ref());
            b.iter(|| {
                let color_map = WuU8x3::run_indexed_image_counts(&image, BinnerU8x3::rgb())
                    .unwrap()
                    .color_map(K);

                image.map_to_image(color_map)
            })
        },
    )
}

fn wu_srgb_palette_par(c: &mut Criterion) {
    bench(c, "wu_srgb_palette_par", benchmark_images(), |b, image| {
        b.iter(|| {
            WuU8x3::run_image_par(image.as_ref(), BinnerU8x3::rgb())
                .unwrap()
                .palette(K)
        })
    })
}

fn wu_srgb_dedup_palette_par(c: &mut Criterion) {
    bench(
        c,
        "wu_srgb_dedup_palette_par",
        benchmark_images(),
        |b, image| {
            let image = dedup::dedup_image_u8_3_counts_par(image.as_ref());
            b.iter(|| {
                WuU8x3::run_indexed_image_counts_par(&image, BinnerU8x3::rgb())
                    .unwrap()
                    .palette(K)
            })
        },
    )
}

fn wu_srgb_remap_par(c: &mut Criterion) {
    bench(c, "wu_srgb_remap_par", benchmark_images(), |b, image| {
        b.iter(|| {
            let color_map = WuU8x3::run_image(image.as_ref(), BinnerU8x3::rgb())
                .unwrap()
                .parallel_color_map(K);

            image.map_to_image(color_map)
        })
    })
}

fn wu_srgb_dedup_remap_par(c: &mut Criterion) {
    bench(
        c,
        "wu_srgb_dedup_remap_par",
        benchmark_images(),
        |b, image| {
            let image = dedup::dedup_image_u8_3_counts_par(image.as_ref());
            b.iter(|| {
                let color_map = WuU8x3::run_indexed_image_counts_par(&image, BinnerU8x3::rgb())
                    .unwrap()
                    .parallel_color_map(K);

                image.map_to_image_par(color_map)
            })
        },
    )
}

fn wu_oklab_palette_single(c: &mut Criterion) {
    bench(
        c,
        "wu_oklab_palette_single",
        benchmark_images(),
        |b, image| {
            let image = image.map_ref(srgb8_to_oklab_par);
            b.iter(|| {
                WuF32x3::run_image(image.as_ref(), BinnerF32x3::oklab_from_srgb8())
                    .unwrap()
                    .palette(K)
            })
        },
    )
}

fn wu_oklab_dedup_palette_single(c: &mut Criterion) {
    bench(
        c,
        "wu_oklab_dedup_palette_single",
        benchmark_counts(),
        |b, image| {
            b.iter(|| {
                WuF32x3::run_indexed_image_counts(image, BinnerF32x3::oklab_from_srgb8())
                    .unwrap()
                    .palette(K)
            })
        },
    )
}

fn wu_oklab_remap_single(c: &mut Criterion) {
    bench(
        c,
        "wu_oklab_remap_single",
        benchmark_images(),
        |b, image| {
            let image = image.map_ref(srgb8_to_oklab_par);
            b.iter(|| {
                let color_map = WuF32x3::run_image(image.as_ref(), BinnerF32x3::oklab_from_srgb8())
                    .unwrap()
                    .color_map(K);

                image.map_to_image(color_map)
            })
        },
    )
}

fn wu_oklab_dedup_remap_single(c: &mut Criterion) {
    bench(
        c,
        "wu_oklab_dedup_remap_single",
        benchmark_counts(),
        |b, image| {
            b.iter(|| {
                let color_map =
                    WuF32x3::run_indexed_image_counts(image, BinnerF32x3::oklab_from_srgb8())
                        .unwrap()
                        .color_map(K);

                image.map_to_image(color_map)
            })
        },
    )
}

fn wu_oklab_palette_par(c: &mut Criterion) {
    bench(c, "wu_oklab_palette_par", benchmark_images(), |b, image| {
        let image = image.map_ref(srgb8_to_oklab_par);
        b.iter(|| {
            WuF32x3::run_image_par(image.as_ref(), BinnerF32x3::oklab_from_srgb8())
                .unwrap()
                .palette(K)
        })
    })
}

fn wu_oklab_dedup_palette_par(c: &mut Criterion) {
    bench(
        c,
        "wu_oklab_dedup_palette_par",
        benchmark_counts(),
        |b, image| {
            b.iter(|| {
                WuF32x3::run_indexed_image_counts_par(image, BinnerF32x3::oklab_from_srgb8())
                    .unwrap()
                    .palette(K)
            })
        },
    )
}

fn wu_oklab_remap_par(c: &mut Criterion) {
    bench(c, "wu_oklab_remap_par", benchmark_counts(), |b, image| {
        b.iter(|| {
            let color_map =
                WuF32x3::run_indexed_image_counts_par(image, BinnerF32x3::oklab_from_srgb8())
                    .unwrap()
                    .parallel_color_map(K);

            image.map_to_image_par(color_map)
        })
    })
}

fn wu_oklab_dedup_remap_par(c: &mut Criterion) {
    bench(
        c,
        "wu_oklab_dedup_remap_par",
        benchmark_counts(),
        |b, image| {
            b.iter(|| {
                let color_map =
                    WuF32x3::run_indexed_image_counts_par(image, BinnerF32x3::oklab_from_srgb8())
                        .unwrap()
                        .parallel_color_map(K);

                image.map_to_image_par(color_map)
            })
        },
    )
}

criterion_group!(
    benches,
    wu_srgb_palette_single,
    wu_srgb_dedup_palette_single,
    wu_srgb_remap_single,
    wu_srgb_dedup_remap_single,
    wu_srgb_palette_par,
    wu_srgb_dedup_palette_par,
    wu_srgb_remap_par,
    wu_srgb_dedup_remap_par,
    wu_oklab_palette_single,
    wu_oklab_dedup_palette_single,
    wu_oklab_remap_single,
    wu_oklab_dedup_remap_single,
    wu_oklab_palette_par,
    wu_oklab_dedup_palette_par,
    wu_oklab_remap_par,
    wu_oklab_dedup_remap_par,
);
criterion_main!(benches);
