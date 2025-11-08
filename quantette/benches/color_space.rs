#[path = "../util/mod.rs"]
mod util;

use criterion::{
    Bencher, BenchmarkId, Criterion, SamplingMode, criterion_group, criterion_main,
    measurement::WallTime,
};
use palette::Srgb;
use quantette::{
    ImageBuf,
    color_space::{oklab_to_srgb8, oklab_to_srgb8_par, srgb8_to_oklab, srgb8_to_oklab_par},
};
use std::time::Duration;
use util::benchmark_images;

fn bench(
    c: &mut Criterion,
    group: &str,
    counts: &[(String, ImageBuf<Srgb<u8>>)],
    mut f: impl FnMut(&mut Bencher<'_, WallTime>, &ImageBuf<Srgb<u8>>),
) {
    let mut group = c.benchmark_group(group);
    group
        .sample_size(30)
        .noise_threshold(0.05)
        .sampling_mode(SamplingMode::Flat)
        .warm_up_time(Duration::from_secs(2));

    for (path, image) in counts {
        group.bench_with_input(BenchmarkId::from_parameter(path), image, &mut f);
    }
}

fn colorspace_srgb8_to_oklab_single(c: &mut Criterion) {
    bench(
        c,
        "colorspace_srgb8_to_oklab_single",
        benchmark_images(),
        |b, image| b.iter(|| srgb8_to_oklab(image.as_slice())),
    )
}

fn colorspace_srgb8_to_oklab_par(c: &mut Criterion) {
    bench(
        c,
        "colorspace_srgb8_to_oklab_par",
        benchmark_images(),
        |b, image| b.iter(|| srgb8_to_oklab_par(image.as_slice())),
    )
}

fn colorspace_oklab_to_srgb8_single(c: &mut Criterion) {
    bench(
        c,
        "colorspace_oklab_to_srgb8_single",
        benchmark_images(),
        |b, image| {
            let oklab = srgb8_to_oklab_par(image.as_slice());
            b.iter(|| oklab_to_srgb8(&oklab))
        },
    )
}

fn colorspace_oklab_to_srgb8_par(c: &mut Criterion) {
    bench(
        c,
        "colorspace_oklab_to_srgb8_par",
        benchmark_images(),
        |b, image| {
            let oklab = srgb8_to_oklab_par(image.as_slice());
            b.iter(|| oklab_to_srgb8_par(&oklab))
        },
    )
}

criterion_group!(
    benches,
    colorspace_srgb8_to_oklab_single,
    colorspace_srgb8_to_oklab_par,
    colorspace_oklab_to_srgb8_single,
    colorspace_oklab_to_srgb8_par,
);
criterion_main!(benches);
