//! The color space used to quantize colors with transparency.
//!
//! A color with alpha is represented by how it looks when composited over a black background and
//! over a white background, with each composite in [`Oklab`]. So, the euclidean distance between
//! two colors measures how differently they appear on both dark and light backgrounds. This
//! accounts for the fact that small differences in alpha can be quite visible. For example, a
//! nearly transparent color over a dark background is noticeably different from a fully
//! transparent one.
//!
//! Compositing is linear in linear sRGB, so a color can be recovered from its two composites.

use crate::color_space::lin_srgb_to_oklab;
use alloc::vec::Vec;
use bytemuck::Zeroable;
use core::array;
use palette::{IntoColor as _, LinSrgb, Oklab, Srgb, Srgba, cast::ArrayCast};
use wide::f32x8;

/// A color represented by its composites over black and white, in [`Oklab`].
#[derive(Debug, Clone, Copy, PartialEq, ArrayCast, Zeroable)]
#[repr(C)]
pub(super) struct Composites {
    /// The `l` component of the composite over black.
    black_l: f32,
    /// The `a` component of the composite over black.
    black_a: f32,
    /// The `b` component of the composite over black.
    black_b: f32,
    /// The `l` component of the composite over white.
    white_l: f32,
    /// The `a` component of the composite over white.
    white_a: f32,
    /// The `b` component of the composite over white.
    white_b: f32,
}

impl Composites {
    /// Returns the lightness of the color averaged over black and white backgrounds.
    #[inline]
    pub(super) fn lightness(self) -> f32 {
        0.5 * (self.black_l + self.white_l)
    }

    /// Projects the color to 4 dimensions: the lightness over black, the lightness over white,
    /// and the mean `a` and `b` components.
    ///
    /// This keeps the dimensions that matter most for transparency (the two lightnesses, which
    /// together encode alpha) while making the color small enough for a 4D histogram.
    #[inline]
    pub(super) fn to_projection(self) -> [f32; 4] {
        [
            self.black_l,
            self.white_l,
            0.5 * (self.black_a + self.white_a),
            0.5 * (self.black_b + self.white_b),
        ]
    }

    /// Creates a color from a projection made by [`to_projection`](Self::to_projection),
    /// using the same `a` and `b` components for both composites.
    #[inline]
    pub(super) fn from_projection([black_l, white_l, a, b]: [f32; 4]) -> Self {
        Self {
            black_l,
            black_a: a,
            black_b: b,
            white_l,
            white_a: a,
            white_b: b,
        }
    }

    /// Returns the composites with both halves replaced by their mean.
    ///
    /// This is the nearest color whose composites are the same over black and white,
    /// i.e., the nearest fully opaque color.
    #[inline]
    pub(super) fn to_opaque(self) -> Self {
        let l = 0.5 * (self.black_l + self.white_l);
        let a = 0.5 * (self.black_a + self.white_a);
        let b = 0.5 * (self.black_b + self.white_b);
        Self {
            black_l: l,
            black_a: a,
            black_b: b,
            white_l: l,
            white_a: a,
            white_b: b,
        }
    }
}

/// Converts up to 8 [`Srgba<u8>`] colors to [`Composites`].
///
/// Only the first `input.len()` elements of the returned array are meaningful.
#[inline]
fn chunk_to_composites(input: &[Srgba<u8>]) -> [Composites; 8] {
    debug_assert!(input.len() <= 8);
    let mut rgb = [[0.0; 8]; 3];
    let mut alpha = [0.0; 8];
    for (i, srgba) in input.iter().enumerate() {
        let lin: [f32; 3] = palette::cast::into_array(srgba.color.into_linear());
        for (c, v) in rgb.iter_mut().zip(lin) {
            c[i] = v;
        }
        alpha[i] = f32::from(srgba.alpha) / f32::from(u8::MAX);
    }

    let alpha = f32x8::new(alpha);
    let [r, g, b] = rgb.map(|c| f32x8::new(c) * alpha);
    let background = f32x8::ONE - alpha;
    let black = lin_srgb_to_oklab(LinSrgb::new(r, g, b));
    let white = lin_srgb_to_oklab(LinSrgb::new(r + background, g + background, b + background));

    let [black_l, black_a, black_b, white_l, white_a, white_b] =
        [black.l, black.a, black.b, white.l, white.a, white.b].map(f32x8::to_array);
    array::from_fn(|i| Composites {
        black_l: black_l[i],
        black_a: black_a[i],
        black_b: black_b[i],
        white_l: white_l[i],
        white_a: white_a[i],
        white_b: white_b[i],
    })
}

/// Converts [`Srgba<u8>`] colors to [`Composites`].
pub(super) fn srgba8_to_composites(input: &[Srgba<u8>], parallel: bool) -> Vec<Composites> {
    let (chunks, remainder) = input.as_chunks::<8>();
    let mut output = par_or_serial!(
        parallel,
        {
            use rayon::prelude::*;
            let mut output = Vec::with_capacity(chunks.len());
            chunks
                .par_iter()
                .map(|chunk| chunk_to_composites(chunk))
                .collect_into_vec(&mut output);
            output
        },
        chunks
            .iter()
            .map(|chunk| chunk_to_composites(chunk))
            .collect::<Vec<_>>(),
    )
    .into_flattened();
    output.extend_from_slice(&chunk_to_composites(remainder)[..remainder.len()]);
    output
}

/// Converts [`Composites`] back to a [`Srgba<u8>`] color.
///
/// If `opaque` is `true`, the alpha is set to `u8::MAX`. Fully transparent results are normalized
/// to `Srgba::new(0, 0, 0, 0)`.
pub(super) fn composites_to_srgba8(color: Composites, opaque: bool) -> Srgba<u8> {
    let black: LinSrgb = Oklab::new(color.black_l, color.black_a, color.black_b).into_color();
    if opaque {
        let srgb: Srgb<u8> = Srgb::from_linear(black);
        return Srgba::new(srgb.red, srgb.green, srgb.blue, u8::MAX);
    }

    let white: LinSrgb = Oklab::new(color.white_l, color.white_a, color.white_b).into_color();
    // `white - black` is `1 - alpha` in each channel.
    let alpha = (1.0
        - ((white.red - black.red) + (white.green - black.green) + (white.blue - black.blue))
            / 3.0)
        .clamp(0.0, 1.0);
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "clamped to the u8 range"
    )]
    let alpha_u8 = libm::roundf(alpha * f32::from(u8::MAX)) as u8;
    if alpha_u8 == 0 {
        return Srgba::new(0, 0, 0, 0);
    }
    let straight = LinSrgb::new(black.red, black.green, black.blue) / alpha;
    let straight = LinSrgb::new(
        straight.red.clamp(0.0, 1.0),
        straight.green.clamp(0.0, 1.0),
        straight.blue.clamp(0.0, 1.0),
    );
    let srgb: Srgb<u8> = Srgb::from_linear(straight);
    Srgba::new(srgb.red, srgb.green, srgb.blue, alpha_u8)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip() {
        let colors = [
            Srgba::new(255, 0, 0, 255),
            Srgba::new(10, 200, 30, 128),
            Srgba::new(0, 0, 255, 1),
            Srgba::new(0, 0, 0, 0),
            Srgba::new(255, 255, 255, 255),
            Srgba::new(77, 88, 99, 200),
        ];
        let composites = srgba8_to_composites(&colors, false);
        for (&color, &c) in colors.iter().zip(&composites) {
            let back = composites_to_srgba8(c, color.alpha == u8::MAX);
            assert_eq!(back.alpha, color.alpha);
            if color.alpha >= 128 {
                for (a, b) in [
                    (back.red, color.red),
                    (back.green, color.green),
                    (back.blue, color.blue),
                ] {
                    assert!(a.abs_diff(b) <= 1, "{color:?} {back:?}");
                }
            }
        }
    }

    #[test]
    fn opaque_composites_are_equal() {
        let c = srgba8_to_composites(&[Srgba::new(12, 34, 56, 255)], false)[0];
        assert_eq!(c, c.to_opaque());
    }
}
