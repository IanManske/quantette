//! Deduplication of [`Srgba<u8>`] pixels.

use crate::IndexedImageCounts;
use alloc::{vec, vec::Vec};
use palette::Srgba;

/// Returns the sort key for a color.
///
/// All fully transparent colors are given the same key, which is the key of `Srgba::new(0, 0, 0, 0)`.
#[inline]
fn key(color: Srgba<u8>) -> u32 {
    if color.alpha == 0 {
        0
    } else {
        u32::from_be_bytes([color.alpha, color.red, color.green, color.blue])
    }
}

/// Returns the color for a sort key.
#[inline]
fn color(key: u32) -> Srgba<u8> {
    let [alpha, red, green, blue] = key.to_be_bytes();
    Srgba::new(red, green, blue, alpha)
}

/// Returns the byte of `value` at the given bit `shift`.
#[inline]
fn byte(value: u64, shift: u32) -> usize {
    #[expect(clippy::cast_possible_truncation, reason = "extracting a byte")]
    usize::from((value >> shift) as u8)
}

/// Sorts `values` by their upper 32 bits using a stable least significant digit radix sort.
///
/// Byte positions where every value has the same byte (e.g., alpha in an opaque image) are skipped.
fn radix_sort_upper_u32(values: &mut Vec<u64>) {
    let mut buffer = Vec::new();
    for shift in [32, 40, 48, 56] {
        let mut counts = [0usize; 256];
        for &v in &*values {
            counts[byte(v, shift)] += 1;
        }
        if counts.contains(&values.len()) {
            continue;
        }

        let mut sum = 0;
        for count in &mut counts {
            let n = *count;
            *count = sum;
            sum += n;
        }

        buffer.resize(values.len(), 0);
        for &v in &*values {
            let bucket = &mut counts[byte(v, shift)];
            buffer[*bucket] = v;
            *bucket += 1;
        }
        core::mem::swap(values, &mut buffer);
    }
}

/// Deduplicates an image of [`Srgba<u8>`] pixels into an [`IndexedImageCounts`].
///
/// All fully transparent pixels are merged into a single `Srgba::new(0, 0, 0, 0)` palette color.
/// The palette is sorted, so the output is deterministic regardless of `parallel`.
///
/// `width * height` must equal `pixels.len()`, which must be at most [`MAX_PIXELS`](crate::MAX_PIXELS).
pub(super) fn dedup_image(
    pixels: &[Srgba<u8>],
    width: u32,
    height: u32,
    parallel: bool,
) -> IndexedImageCounts<Srgba<u8>, u32> {
    debug_assert_eq!(u64::from(width) * u64::from(height), pixels.len() as u64);

    // Pack the color key and pixel index together so that sorting groups equal colors.
    let mut values = pixels
        .iter()
        .zip(0u32..)
        .map(|(&p, i)| (u64::from(key(p)) << 32) | u64::from(i))
        .collect::<Vec<_>>();

    par_or_serial!(
        parallel,
        {
            use rayon::prelude::*;
            values.par_sort_unstable();
        },
        radix_sort_upper_u32(&mut values),
    );

    let mut palette = Vec::new();
    let mut counts = Vec::<u32>::new();
    let mut indices = vec![0u32; pixels.len()];
    let mut prev = None;
    for v in values {
        #[expect(clippy::cast_possible_truncation, reason = "unpacking two u32s")]
        let (key, i) = ((v >> 32) as u32, v as u32);
        if prev != Some(key) {
            prev = Some(key);
            palette.push(color(key));
            counts.push(0);
        }
        if let Some(count) = counts.last_mut() {
            *count += 1;
        }
        #[expect(clippy::cast_possible_truncation, reason = "at most MAX_PIXELS colors")]
        (indices[i as usize] = (palette.len() - 1) as u32);
    }

    IndexedImageCounts::new_unchecked(width, height, palette, counts, indices)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dedup_matches_input() {
        let pixels = [
            Srgba::new(1, 2, 3, 255),
            Srgba::new(9, 9, 9, 0),
            Srgba::new(1, 2, 3, 255),
            Srgba::new(1, 2, 3, 128),
            Srgba::new(0, 0, 0, 0),
            Srgba::new(255, 0, 0, 255),
        ];
        for parallel in [false, true] {
            let image = dedup_image(&pixels, 3, 2, parallel);
            assert_eq!(image.palette().len(), 4);
            assert_eq!(image.counts().iter().sum::<u32>(), 6);
            for (&pixel, &i) in pixels.iter().zip(image.indices()) {
                let expected = if pixel.alpha == 0 {
                    Srgba::new(0, 0, 0, 0)
                } else {
                    pixel
                };
                assert_eq!(image.palette()[i as usize], expected);
            }
            let palette = image.palette();
            assert!(palette.windows(2).all(|w| key(w[0]) < key(w[1])));
        }
    }

    #[test]
    fn radix_sort_sorts() {
        let mut values = (0..1000u64)
            .map(|i| (((i * 2_654_435_761) % 4_294_967_291) << 32) | i)
            .collect::<Vec<_>>();
        let mut expected = values.clone();
        expected.sort_by_key(|v| v >> 32);
        radix_sort_upper_u32(&mut values);
        assert_eq!(values, expected);
    }
}
