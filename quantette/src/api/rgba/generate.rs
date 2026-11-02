//! Palette generation in the [`Composites`] color space.

use super::space::{Composites, composites_to_srgba8, srgba8_to_composites};
#[cfg(feature = "threads")]
use crate::color_map::NearestNeighborParallelColorMap;
use crate::{
    IndexedColorMap as _, PaletteBuf, PaletteSize,
    color_map::NearestNeighborColorMap,
    wu::{
        BinnerF32x3,
        shared::{Histogram4, Stats, Wu, sum_of_squares},
    },
};
use alloc::{vec, vec::Vec};
use palette::{
    Srgba,
    cast::{self, ArrayCast},
};

/// The alpha class of a color, used to make sure that fully transparent and fully opaque pixels
/// stay that way.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Class {
    /// A partially transparent color, or a palette color that may be used for any pixel.
    Free,
    /// A fully opaque color.
    Opaque,
    /// A fully transparent color.
    Transparent,
}

impl Class {
    /// Returns the class of a color with the given alpha value.
    #[inline]
    pub(super) const fn of(alpha: u8) -> Self {
        match alpha {
            0 => Self::Transparent,
            u8::MAX => Self::Opaque,
            _ => Self::Free,
        }
    }
}

/// Returns the [`Composites`] for a fully transparent color.
fn transparent() -> Composites {
    srgba8_to_composites(&[Srgba::new(0, 0, 0, 0)], false)[0]
}

/// Returns the squared euclidean distance between two colors.
#[inline]
fn distance(a: Composites, b: Composites) -> f32 {
    let a: [f32; 6] = cast::into_array(a);
    let b: [f32; 6] = cast::into_array(b);
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// Returns the index of the color in `palette` nearest to `color`
/// among those for which `filter` returns `true`.
pub(super) fn nearest_where(
    palette: &[Composites],
    color: Composites,
    filter: impl Fn(usize) -> bool,
) -> Option<u8> {
    let mut nearest = None;
    let mut min_distance = f32::INFINITY;
    for (i, &c) in palette.iter().enumerate() {
        if filter(i) {
            let d = distance(c, color);
            if d < min_distance {
                min_distance = d;
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "palette has at most 256 colors"
                )]
                (nearest = Some(i as u8));
            }
        }
    }
    nearest
}

/// Maps each color to the index of its nearest palette color, subject to the following constraints:
/// - Fully transparent colors are mapped to a fully transparent palette color, if there is one.
/// - Fully opaque colors are mapped to the nearest fully opaque palette color, if there is one.
///
/// This ensures that transparent regions stay transparent and opaque regions stay opaque.
pub(super) fn assign_constrained(
    colors: &[Composites],
    color_classes: &[Class],
    palette: &[Composites],
    palette_classes: &[Class],
    parallel: bool,
) -> Vec<u8> {
    let palette_buf = PaletteBuf::new_unchecked(palette.to_vec());
    let mut indices = par_or_serial!(
        parallel,
        NearestNeighborParallelColorMap::new(palette_buf).map_to_indices(colors),
        NearestNeighborColorMap::new(palette_buf).map_to_indices(colors),
    );

    let transparent = palette_classes
        .iter()
        .position(|&c| c == Class::Transparent);
    let has_opaque = palette_classes.contains(&Class::Opaque);

    let fix = |((index, &color), &class): ((&mut u8, &Composites), &Class)| match class {
        Class::Transparent => {
            if let Some(i) = transparent {
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "palette has at most 256 colors"
                )]
                (*index = i as u8);
            }
        }
        Class::Opaque if has_opaque && palette_classes[usize::from(*index)] != Class::Opaque => {
            if let Some(i) = nearest_where(palette, color, |i| palette_classes[i] == Class::Opaque)
            {
                *index = i;
            }
        }
        Class::Opaque | Class::Free => {}
    };

    par_or_serial!(
        parallel,
        {
            use rayon::prelude::*;
            indices
                .par_iter_mut()
                .zip(colors)
                .zip(color_classes)
                .for_each(fix);
        },
        indices
            .iter_mut()
            .zip(colors)
            .zip(color_classes)
            .for_each(fix),
    );
    indices
}

/// Makes sure that the centroids can satisfy the constraints of [`assign_constrained`].
///
/// - If any color is fully opaque but no centroid is, then the centroid nearest to being fully
///   opaque is made fully opaque.
/// - If any color is fully transparent but no centroid is, then a fully transparent centroid is
///   added if there is room. Otherwise, the centroid nearest to it is replaced (unless that
///   centroid is the only fully opaque one).
pub(super) fn ensure_alpha_classes(
    centroids: &mut Vec<Composites>,
    classes: &mut Vec<Class>,
    color_classes: &[Class],
    k: PaletteSize,
) {
    for (c, &class) in centroids.iter_mut().zip(&*classes) {
        match class {
            Class::Opaque => *c = c.to_opaque(),
            Class::Transparent => *c = transparent(),
            Class::Free => {}
        }
    }

    if color_classes.contains(&Class::Opaque) && !classes.contains(&Class::Opaque) {
        let nearest = centroids
            .iter()
            .enumerate()
            .filter(|&(i, _)| classes[i] == Class::Free)
            .min_by(|(_, a), (_, b)| {
                distance(**a, a.to_opaque()).total_cmp(&distance(**b, b.to_opaque()))
            })
            .map(|(i, _)| i);
        if let Some(i) = nearest {
            centroids[i] = centroids[i].to_opaque();
            classes[i] = Class::Opaque;
        }
    }

    if color_classes.contains(&Class::Transparent) && !classes.contains(&Class::Transparent) {
        let transparent = transparent();
        if centroids.len() < k.as_usize() {
            centroids.push(transparent);
            classes.push(Class::Transparent);
        } else {
            let num_opaque = classes.iter().filter(|&&c| c == Class::Opaque).count();
            let nearest = nearest_where(centroids, transparent, |i| {
                classes[i] != Class::Opaque || num_opaque > 1
            });
            if let Some(i) = nearest {
                let i = usize::from(i);
                centroids[i] = transparent;
                classes[i] = Class::Transparent;
            }
        }
    }
}

/// Converts the final centroids to a [`Srgba<u8>`] palette, removing any duplicate colors.
pub(super) fn finalize_palette(
    centroids: &[Composites],
    classes: &[Class],
) -> PaletteBuf<Srgba<u8>> {
    let mut palette = Vec::with_capacity(centroids.len());
    for (&c, &class) in centroids.iter().zip(classes) {
        let color = match class {
            Class::Transparent => Srgba::new(0, 0, 0, 0),
            Class::Opaque => composites_to_srgba8(c, true),
            Class::Free => composites_to_srgba8(c, false),
        };
        if !palette.contains(&color) {
            palette.push(color);
        }
    }
    PaletteBuf::new_unchecked(palette)
}

/// Returns a weighted sample of at most `n` colors as `(index, weight)` pairs, where colors are
/// chosen with probability proportional to their `weights`.
///
/// Systematic sampling is used, so the result is deterministic, and the total weight is preserved.
/// If there are at most `n` colors, then all colors are returned as is.
pub(super) fn weighted_sample(weights: &[f64], n: usize) -> Vec<(usize, f64)> {
    if weights.len() <= n {
        return weights.iter().copied().enumerate().collect();
    }
    let total = weights.iter().sum::<f64>();
    #[expect(clippy::cast_precision_loss, reason = "n is small")]
    let step = total / n as f64;
    let mut sample = Vec::<(usize, f64)>::with_capacity(n);
    let mut next = 0.5 * step;
    let mut cumulative = 0.0;
    for (i, &w) in weights.iter().enumerate() {
        cumulative += w;
        while next < cumulative {
            match sample.last_mut() {
                Some((j, weight)) if *j == i => *weight += step,
                _ => sample.push((i, step)),
            }
            next += step;
        }
    }
    sample
}

#[cfg(feature = "kmeans")]
/// Returns at most `n` distinct colors as `(index, weight)` pairs, keeping their original weights.
///
/// Unlike [`weighted_sample`], colors are chosen with probability proportional to the square root
/// of their weights. This keeps rare but distinct colors (e.g., anti-aliased edges in a flat logo)
/// as candidates for initial centroids, rather than mostly choosing the most common colors.
fn diverse_sample(weights: &[f64], n: usize) -> Vec<(usize, f64)> {
    if weights.len() <= n {
        return weights.iter().copied().enumerate().collect();
    }
    let total = weights.iter().map(|&w| libm::sqrt(w)).sum::<f64>();
    #[expect(clippy::cast_precision_loss, reason = "n is small")]
    let step = total / n as f64;
    let mut sample = Vec::with_capacity(n);
    let mut next = 0.5 * step;
    let mut cumulative = 0.0;
    for (i, &w) in weights.iter().enumerate() {
        cumulative += libm::sqrt(w);
        if next < cumulative {
            sample.push((i, w));
            while next < cumulative {
                next += step;
            }
        }
    }
    sample
}

/// Refines `centroids` using Lloyd's algorithm on the weighted `colors`.
///
/// Colors are assigned using [`assign_constrained`]. Then, each fully opaque centroid is moved to
/// the nearest fully opaque color to the mean of its assigned colors, and each other centroid
/// (except for the fully transparent one) is moved to the mean of its assigned colors.
pub(super) fn refine_centroids(
    centroids: &mut [Composites],
    classes: &[Class],
    colors: &[Composites],
    color_classes: &[Class],
    weights: &[f64],
    max_iterations: usize,
    parallel: bool,
) {
    /// Stop once no centroid moves more than this (squared) distance in an iteration.
    const MIN_SHIFT: f64 = 1e-10;
    let mut assignments = Vec::new();
    for _ in 0..max_iterations {
        let new_assignments =
            assign_constrained(colors, color_classes, centroids, classes, parallel);
        if new_assignments == assignments {
            break;
        }
        assignments = new_assignments;
        let mut sums = vec![([0.0f64; 6], 0.0f64); centroids.len()];
        for ((&color, &w), &i) in colors.iter().zip(weights).zip(&assignments) {
            let (sum, total) = &mut sums[usize::from(i)];
            let color: [f32; 6] = cast::into_array(color);
            for (s, c) in sum.iter_mut().zip(color) {
                *s += w * f64::from(c);
            }
            *total += w;
        }
        let mut max_shift = 0.0f64;
        for ((centroid, &class), (sum, total)) in centroids.iter_mut().zip(classes).zip(sums) {
            if total <= 0.0 || class == Class::Transparent {
                continue;
            }
            #[expect(clippy::cast_possible_truncation, reason = "f64 to f32 is intentional")]
            let mut new: Composites = cast::from_array(sum.map(|s| (s / total) as f32));
            if class == Class::Opaque {
                new = new.to_opaque();
            }
            max_shift = max_shift.max(f64::from(distance(*centroid, new)));
            *centroid = new;
        }
        if max_shift <= MIN_SHIFT {
            break;
        }
    }
}

#[cfg(feature = "kmeans")]
/// Returns the total weighted squared distance from each color to the palette color it is
/// assigned to by [`assign_constrained`].
fn objective(
    centroids: &[Composites],
    classes: &[Class],
    colors: &[Composites],
    color_classes: &[Class],
    weights: &[f64],
    parallel: bool,
) -> f64 {
    let assignments = assign_constrained(colors, color_classes, centroids, classes, parallel);
    colors
        .iter()
        .zip(weights)
        .zip(assignments)
        .map(|((&c, &w), i)| w * f64::from(distance(c, centroids[usize::from(i)])))
        .sum()
}

/// A set of colors with their alpha classes and weights.
pub(super) struct Weighted {
    /// The colors.
    pub(super) colors: Vec<Composites>,
    /// The alpha class of each color.
    pub(super) classes: Vec<Class>,
    /// The weight of each color.
    pub(super) weights: Vec<f64>,
}

impl Weighted {
    /// Returns a weighted sample of at most `n` of the given colors (see [`weighted_sample`]).
    pub(super) fn sample(
        colors: &[Composites],
        classes: &[Class],
        weights: &[f64],
        n: usize,
    ) -> Self {
        let sample = weighted_sample(weights, n);
        Self {
            colors: sample.iter().map(|&(i, _)| colors[i]).collect(),
            classes: sample.iter().map(|&(i, _)| classes[i]).collect(),
            weights: sample.iter().map(|&(_, w)| w).collect(),
        }
    }

    /// Refines `centroids` on these colors using [`refine_centroids`].
    fn refine(
        &self,
        centroids: &mut [Composites],
        classes: &[Class],
        iterations: usize,
        parallel: bool,
    ) {
        refine_centroids(
            centroids,
            classes,
            &self.colors,
            &self.classes,
            &self.weights,
            iterations,
            parallel,
        );
    }

    /// Returns the [`objective`] for `centroids` on these colors.
    #[cfg(feature = "kmeans")]
    fn objective(&self, centroids: &[Composites], classes: &[Class], parallel: bool) -> f64 {
        objective(
            centroids,
            classes,
            &self.colors,
            &self.classes,
            &self.weights,
            parallel,
        )
    }
}

#[cfg(feature = "kmeans")]
/// Returns refined centroids from k-means on the given `sample` of the colors.
///
/// k-means is sensitive to its initial centroids, especially for small palettes. So, for small
/// palettes, k-means is run a few times with different seeds, and the centroids with the lowest
/// error are returned.
pub(super) fn kmeans_centroids(
    sample: &Weighted,
    color_classes: &[Class],
    k: PaletteSize,
    seed: u64,
    parallel: bool,
) -> (Vec<Composites>, Vec<Class>) {
    /// The maximum number of colors to use for k-means++.
    const SEED_SAMPLES: usize = 16384;
    /// The maximum number of iterations of Lloyd's algorithm.
    const ITERATIONS: usize = 16;

    let seeds = diverse_sample(&sample.weights, SEED_SAMPLES);
    let seeds = Weighted {
        colors: seeds.iter().map(|&(i, _)| sample.colors[i]).collect(),
        classes: seeds.iter().map(|&(i, _)| sample.classes[i]).collect(),
        weights: seeds.iter().map(|&(_, w)| w).collect(),
    };
    let restarts = (128 / k.as_usize()).clamp(1, 4);

    let mut best = None;
    for restart in 0..restarts {
        let (mut centroids, mut classes) = kmeans_plus_plus(
            &seeds.colors,
            &seeds.classes,
            &seeds.weights,
            k,
            seed.wrapping_add(restart as u64),
            parallel,
        );
        add_farthest_colors(
            &mut centroids,
            &mut classes,
            &sample.colors,
            &sample.classes,
            k,
        );
        ensure_alpha_classes(&mut centroids, &mut classes, color_classes, k);
        sample.refine(&mut centroids, &classes, ITERATIONS, parallel);
        let error = sample.objective(&centroids, &classes, parallel);
        if best.as_ref().is_none_or(|&(e, _, _)| error < e) {
            best = Some((error, centroids, classes));
        }
    }
    best.map(|(_, centroids, classes)| (centroids, classes))
        .unwrap_or_default()
}

#[cfg(feature = "kmeans")]
/// Returns the colors at the given `targets` in the cumulative sum of weighted distances.
///
/// That is, colors are chosen with probability proportional to their weighted distance.
fn sample_candidates(
    colors: &[Composites],
    distances: &[f64],
    weights: &[f64],
    mut targets: Vec<f64>,
) -> Vec<Composites> {
    targets.sort_by(f64::total_cmp);
    let mut chosen = Vec::with_capacity(targets.len());
    let mut cumulative = 0.0;
    let mut targets = targets.into_iter().peekable();
    for ((&color, d), w) in colors.iter().zip(distances).zip(weights) {
        cumulative += d * w;
        while targets.next_if(|&t| t <= cumulative).is_some() {
            chosen.push(color);
        }
    }
    // Floating point error may leave targets past the end.
    if let Some(&last) = colors.last() {
        chosen.extend(targets.map(|_| last));
    }
    chosen
}

#[cfg(feature = "kmeans")]
/// Returns the total weighted distance (the potential) that would result from adding each of the
/// `candidates` as a centroid, computed in a single pass over the colors.
fn candidate_potentials(
    colors: &[Composites],
    distances: &[f64],
    weights: &[f64],
    candidates: &[Composites],
    parallel: bool,
) -> Vec<f64> {
    /// The number of colors processed together. Sums are computed per chunk and then added in
    /// order, so the result is the same whether or not it is computed in parallel.
    const CHUNK: usize = 1024;

    let chunk_potentials = |(chunk, (distances, weights)): (&[Composites], (&[f64], &[f64]))| {
        let mut sums = vec![0.0; candidates.len()];
        for ((&color, &d), &w) in chunk.iter().zip(distances).zip(weights) {
            for (sum, &candidate) in sums.iter_mut().zip(candidates) {
                *sum += d.min(f64::from(distance(color, candidate))) * w;
            }
        }
        sums
    };
    let per_chunk = par_or_serial!(
        parallel,
        {
            use rayon::prelude::*;
            colors
                .par_chunks(CHUNK)
                .zip(distances.par_chunks(CHUNK).zip(weights.par_chunks(CHUNK)))
                .map(chunk_potentials)
                .collect::<Vec<_>>()
        },
        colors
            .chunks(CHUNK)
            .zip(distances.chunks(CHUNK).zip(weights.chunks(CHUNK)))
            .map(chunk_potentials)
            .collect::<Vec<_>>(),
    );

    let mut potentials = vec![0.0; candidates.len()];
    for sums in per_chunk {
        for (p, s) in potentials.iter_mut().zip(sums) {
            *p += s;
        }
    }
    potentials
}

#[cfg(feature = "kmeans")]
/// Chooses initial centroids from the weighted `colors` using greedy k-means++.
///
/// Each new centroid is chosen from several candidates sampled with probability proportional to
/// their weighted squared distance from the existing centroids, keeping the candidate that lowers
/// the total weighted squared distance the most. The centroids keep the class of their color.
pub(super) fn kmeans_plus_plus(
    colors: &[Composites],
    color_classes: &[Class],
    weights: &[f64],
    k: PaletteSize,
    seed: u64,
    parallel: bool,
) -> (Vec<Composites>, Vec<Class>) {
    use rand::{RngExt as _, SeedableRng as _};
    use rand_xoshiro::Xoroshiro128PlusPlus;

    let mut rng = Xoroshiro128PlusPlus::seed_from_u64(seed);
    let k = k.as_usize().min(colors.len());
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss,
        reason = "small positive number"
    )]
    let trials = 2 + libm::log(k as f64) as usize;

    let Some(first) = weights
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(i, _)| i)
    else {
        return (Vec::new(), Vec::new());
    };

    let mut centroids = vec![colors[first]];
    let mut classes = vec![color_classes[first]];
    let mut min_distances = colors
        .iter()
        .map(|&c| f64::from(distance(c, colors[first])))
        .collect::<Vec<_>>();

    while centroids.len() < k {
        let potential = min_distances
            .iter()
            .zip(weights)
            .map(|(d, w)| d * w)
            .sum::<f64>();
        if potential <= 0.0 {
            break;
        }

        let targets = (0..trials)
            .map(|_| rng.random::<f64>() * potential)
            .collect::<Vec<_>>();
        let chosen = sample_candidates(colors, &min_distances, weights, targets);
        let potentials = candidate_potentials(colors, &min_distances, weights, &chosen, parallel);

        let Some(best) = potentials
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(i, _)| chosen[i])
        else {
            break;
        };
        let best_index = colors.iter().position(|&c| c == best).unwrap_or(0);
        centroids.push(best);
        classes.push(color_classes[best_index]);

        let update = |(d, &c): (&mut f64, &Composites)| *d = d.min(f64::from(distance(c, best)));
        par_or_serial!(
            parallel,
            {
                use rayon::prelude::*;
                min_distances.par_iter_mut().zip(colors).for_each(update);
            },
            min_distances.iter_mut().zip(colors).for_each(update),
        );
    }

    (centroids, classes)
}

/// Adds colors to `centroids` until there are `k` centroids or no more distinct colors.
///
/// Wu's method cannot split a histogram bin, so it may return fewer than `k` colors for images
/// whose colors are concentrated in a few bins (e.g., faint shadows with many low alpha levels).
/// So, this greedily adds the colors that are the farthest from all existing centroids.
/// The added centroids keep the class of their color.
pub(super) fn add_farthest_colors(
    centroids: &mut Vec<Composites>,
    classes: &mut Vec<Class>,
    colors: &[Composites],
    color_classes: &[Class],
    k: PaletteSize,
) {
    if centroids.len() >= k.as_usize() || colors.is_empty() {
        return;
    }

    let mut min_distances = colors
        .iter()
        .map(|&color| {
            centroids
                .iter()
                .map(|&c| distance(color, c))
                .fold(f32::INFINITY, f32::min)
        })
        .collect::<Vec<_>>();

    while centroids.len() < k.as_usize() {
        let Some((farthest, &max_distance)) = min_distances
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
        else {
            break;
        };
        if max_distance <= 0.0 {
            break;
        }
        let new = colors[farthest];
        centroids.push(new);
        classes.push(color_classes[farthest]);
        for (d, &color) in min_distances.iter_mut().zip(colors) {
            *d = d.min(distance(color, new));
        }
    }
}

/// Returns refined centroids from Wu's color quantization method on the unique `colors`.
///
/// Wu's method runs on a 4D projection of the colors (see [`Composites::to_projection`]).
/// Since it cannot split histogram bins, missing centroids are added using
/// [`add_farthest_colors`]. The centroids are then refined on the `sample` of the colors.
pub(super) fn wu_centroids(
    colors: &[Composites],
    color_classes: &[Class],
    weights: &[f64],
    sample: &Weighted,
    k: PaletteSize,
    parallel: bool,
) -> (Vec<Composites>, Vec<Class>) {
    /// Centroids with an alpha at least this high are made fully opaque.
    const OPAQUE_ALPHA: u8 = u8::MAX - 8;
    /// The histogram stores integer counts, so weights are scaled by this to keep precision.
    const WEIGHT_SCALE: f64 = 16.0;
    /// The number of iterations of Lloyd's algorithm.
    const ITERATIONS: usize = 3;

    let projected = colors
        .iter()
        .map(|c| cast::from_array::<Projected>(c.to_projection()))
        .collect::<Vec<_>>();
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "weights are small and positive"
    )]
    let counts = weights
        .iter()
        .map(|&w| (libm::round(w * WEIGHT_SCALE) as u32).max(1))
        .collect::<Vec<_>>();

    let mut centroids = wu_palette(&projected, &counts, k, parallel)
        .iter()
        .map(|&p| Composites::from_projection(cast::into_array(p)))
        .collect::<Vec<_>>();
    let mut classes = centroids
        .iter()
        .map(|&c| match composites_to_srgba8(c, false).alpha {
            0 => Class::Transparent,
            alpha if alpha >= OPAQUE_ALPHA => Class::Opaque,
            _ => Class::Free,
        })
        .collect();

    add_farthest_colors(
        &mut centroids,
        &mut classes,
        &sample.colors,
        &sample.classes,
        k,
    );
    ensure_alpha_classes(&mut centroids, &mut classes, color_classes, k);
    sample.refine(&mut centroids, &classes, ITERATIONS, parallel);
    (centroids, classes)
}

/// A color projected to 4 dimensions for Wu's method (see [`Composites::to_projection`]).
#[derive(Debug, Clone, Copy, ArrayCast)]
#[repr(C)]
struct Projected {
    /// The lightness over black.
    black_l: f32,
    /// The lightness over white.
    white_l: f32,
    /// The mean `a` component.
    a: f32,
    /// The mean `b` component.
    b: f32,
}

/// The number of histogram bins for each projected component used by Wu's quantization method.
const WU_BINS: [usize; 4] = [32, 32, 16, 16];

/// The histogram type for Wu's quantization method on projected colors.
type WuHistogram =
    Histogram4<Stats<f64, 4>, { WU_BINS[0] }, { WU_BINS[1] }, { WU_BINS[2] }, { WU_BINS[3] }>;

/// Returns the flattened histogram bin index for a projected color.
#[inline]
fn wu_bin_index(color: [f32; 4]) -> usize {
    // Compositing and averaging keep the `a` and `b` components inside the `Oklab` ranges.
    let [l, a, b] = BinnerF32x3::<1, 1, 1>::OKLAB_COMPONENT_RANGES_FROM_SRGB8;
    let ranges = [l, l, a, b];
    let mut index = 0;
    for ((c, (min, max)), bins) in color.into_iter().zip(ranges).zip(WU_BINS) {
        #[expect(clippy::cast_precision_loss, reason = "small constant")]
        let bins_f32 = bins as f32;
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "clamped to the range of bins"
        )]
        let bin = ((c - min) * (bins_f32 / (max - min))).clamp(0.0, bins_f32 - 1.0) as usize;
        index = index * bins + bin;
    }
    index
}

/// Adds the given projected colors and their counts to the histogram.
fn wu_add_colors(hist: &mut WuHistogram, colors: &[Projected], counts: &[u32]) {
    let hist = hist.as_flattened_mut();
    for (&color, &n) in colors.iter().zip(counts) {
        let color: [f32; 4] = cast::into_array(color);
        let Stats { count, components, sum_squared } = &mut hist[wu_bin_index(color)];
        let color = color.map(f64::from);
        let w = f64::from(n);
        *count += n;
        for (c, v) in components.iter_mut().zip(color) {
            *c += w * v;
        }
        *sum_squared += w * sum_of_squares(color);
    }
}

/// Runs Wu's color quantization method on projected colors and their counts.
fn wu_palette(
    colors: &[Projected],
    counts: &[u32],
    k: PaletteSize,
    parallel: bool,
) -> PaletteBuf<Projected> {
    let mut hist = par_or_serial!(
        parallel,
        {
            use rayon::prelude::*;
            let num_bins = WU_BINS.iter().product::<usize>();
            let chunk_size = colors
                .len()
                .div_ceil(rayon::current_num_threads())
                .max(num_bins * 4);
            colors
                .par_chunks(chunk_size)
                .zip(counts.par_chunks(chunk_size))
                .map(|(colors, counts)| {
                    let mut hist = WuHistogram::new();
                    wu_add_colors(&mut hist, colors, counts);
                    hist
                })
                .reduce_with(WuHistogram::merge_partial)
                .unwrap_or_else(WuHistogram::new)
        },
        {
            let mut hist = WuHistogram::new();
            wu_add_colors(&mut hist, colors, counts);
            hist
        },
    );
    hist.calc_cumulative_moments();
    Wu::<_, Projected, f32, _>::new(&hist).palette(k)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::rgba::tests::sprite;

    #[test]
    fn weighted_sample_preserves_weight() {
        let weights = (1..=1000).map(f64::from).collect::<Vec<_>>();
        let sample = weighted_sample(&weights, 100);
        assert!(sample.len() <= 100);
        let total = sample.iter().map(|&(_, w)| w).sum::<f64>();
        assert!((total - weights.iter().sum::<f64>()).abs() < 1e-6 * total);
        assert!(sample.windows(2).all(|w| w[0].0 < w[1].0));
    }

    #[test]
    fn wu_histogram_volume_matches_sum() {
        let (_, _, pixels) = sprite();
        let colors = srgba8_to_composites(&pixels, false)
            .into_iter()
            .map(|c| cast::from_array::<Projected>(c.to_projection()))
            .collect::<Vec<_>>();
        let counts = vec![1; colors.len()];
        let mut hist = WuHistogram::new();
        wu_add_colors(&mut hist, &colors, &counts);
        let total = hist.as_flattened_mut().iter().map(|s| s.count).sum::<u32>();
        hist.calc_cumulative_moments();
        let last = *hist.as_flattened_mut().last().unwrap();
        assert_eq!(total, u32::try_from(pixels.len()).unwrap_or(0));
        assert_eq!(last.count, total);
    }
}
