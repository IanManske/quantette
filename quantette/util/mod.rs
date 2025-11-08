#![allow(dead_code)]

use image::RgbImage;
use palette::{Oklab, Srgb};
use quantette::{ImageBuf, IndexedImageCounts, color_space::srgb8_to_oklab_par, dedup};
use std::{
    cmp::Reverse,
    ffi::OsStr,
    fs,
    path::{Path, PathBuf},
    sync::OnceLock,
};

pub fn load_images<P: AsRef<Path>>(images: &[P]) -> Vec<RgbImage> {
    images
        .iter()
        .map(|path| image::open(path).unwrap().into_rgb8())
        .collect()
}

pub fn load_image_dir(dir: impl AsRef<Path>) -> Vec<(PathBuf, RgbImage)> {
    let mut paths = fs::read_dir(dir)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .collect::<Vec<_>>();

    paths.sort_unstable();

    let images = load_images(&paths);

    paths.into_iter().zip(images).collect()
}

fn root_dir() -> PathBuf {
    // assume current exe path is something like: target/profile/dir/current_exe
    std::env::current_exe()
        .unwrap()
        .parent()
        .and_then(Path::parent)
        .and_then(Path::parent)
        .and_then(Path::parent)
        .unwrap()
        .into()
}

pub fn load_image_dir_relative_to_root(dir: impl AsRef<Path>) -> Vec<(PathBuf, RgbImage)> {
    let mut root = root_dir();
    root.push(dir);
    load_image_dir(root)
}

static BENCHMARK_IMAGES: OnceLock<Vec<(String, ImageBuf<Srgb<u8>>)>> = OnceLock::new();

pub fn benchmark_images() -> &'static [(String, ImageBuf<Srgb<u8>>)] {
    BENCHMARK_IMAGES.get_or_init(|| {
        let images = {
            let mut path = root_dir();
            path.push("img");
            path.push("unsplash");
            path.push("img");
            path
        };

        let mut images = fs::read_dir(images)
            .unwrap()
            .map(|dir| {
                let dir = dir.unwrap().path();
                let resolution = dir.file_stem().and_then(OsStr::to_str).unwrap();
                let resolution = if let Some((width, height)) = resolution.split_once('x') {
                    (width.parse().unwrap(), height.parse().unwrap())
                } else {
                    (u32::MAX, u32::MAX)
                };
                (resolution, dir)
            })
            .collect::<Vec<_>>();

        images.sort_unstable_by_key(|&(res, _)| Reverse(res));

        images
            .into_iter()
            .flat_map(|(_, dir)| load_image_dir(dir))
            .map(|(path, img)| {
                let name = path
                    .components()
                    .rev()
                    .take(2)
                    .collect::<Vec<_>>()
                    .into_iter()
                    .rev()
                    .collect::<PathBuf>()
                    .display()
                    .to_string();

                (name, img.try_into().unwrap())
            })
            .collect()
    })
}

fn to_oklab_counts(
    images: &[(String, ImageBuf<Srgb<u8>>)],
) -> Vec<(String, IndexedImageCounts<Oklab, u32>)> {
    images
        .iter()
        .map(|(path, image)| {
            (
                path.clone(),
                dedup::dedup_image_u8_3_counts_par(image.as_ref())
                    .map(|palette| srgb8_to_oklab_par(&palette)),
            )
        })
        .collect()
}

static BENCHMARK_COUNTS: OnceLock<Vec<(String, IndexedImageCounts<Oklab, u32>)>> = OnceLock::new();

pub fn benchmark_counts() -> &'static [(String, IndexedImageCounts<Oklab, u32>)] {
    BENCHMARK_COUNTS.get_or_init(|| to_oklab_counts(benchmark_images()))
}
