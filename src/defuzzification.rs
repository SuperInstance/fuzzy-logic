//! Defuzzification methods.

/// Centroid defuzzification: weighted average of x values by membership.
pub fn centroid(samples: &[(f64, f64)]) -> f64 {
    let num: f64 = samples.iter().map(|(x, mu)| x * mu).sum();
    let den: f64 = samples.iter().map(|(_, mu)| mu).sum();
    if den.abs() < 1e-12 { 0.0 } else { num / den }
}

/// Mean of maxima defuzzification: average of all x values where mu is maximum.
pub fn mean_of_maxima(samples: &[(f64, f64)]) -> f64 {
    let max_mu = samples.iter().map(|(_, mu)| *mu).fold(0.0_f64, f64::max);
    if max_mu < 1e-12 { return 0.0; }
    let maxima: Vec<f64> = samples
        .iter()
        .filter(|(_, mu)| (mu - max_mu).abs() < 1e-9)
        .map(|(x, _)| *x)
        .collect();
    if maxima.is_empty() { 0.0 } else { maxima.iter().sum::<f64>() / maxima.len() as f64 }
}

/// Bisector defuzzification: point that splits the area in half.
pub fn bisector(samples: &[(f64, f64)]) -> f64 {
    let total_area: f64 = samples.iter().map(|(_, mu)| *mu).sum();
    if total_area.abs() < 1e-12 { return 0.0; }

    let half = total_area / 2.0;
    let mut accumulated = 0.0;
    for (x, mu) in samples {
        accumulated += mu;
        if accumulated >= half {
            return *x;
        }
    }
    samples.last().map(|(x, _)| *x).unwrap_or(0.0)
}

/// Weighted average defuzzification (useful for Sugeno).
/// Takes (value, weight) pairs.
pub fn weighted_average(items: &[(f64, f64)]) -> f64 {
    let num: f64 = items.iter().map(|(v, w)| v * w).sum();
    let den: f64 = items.iter().map(|(_, w)| *w).sum();
    if den.abs() < 1e-12 { 0.0 } else { num / den }
}

/// Largest of maximum.
pub fn largest_of_max(samples: &[(f64, f64)]) -> f64 {
    let max_mu = samples.iter().map(|(_, mu)| *mu).fold(0.0_f64, f64::max);
    samples
        .iter()
        .filter(|(_, mu)| (mu - max_mu).abs() < 1e-9)
        .last()
        .map(|(x, _)| *x)
        .unwrap_or(0.0)
}

/// Smallest of maximum.
pub fn smallest_of_max(samples: &[(f64, f64)]) -> f64 {
    let max_mu = samples.iter().map(|(_, mu)| *mu).fold(0.0_f64, f64::max);
    samples
        .iter()
        .filter(|(_, mu)| (mu - max_mu).abs() < 1e-9)
        .next()
        .map(|(x, _)| *x)
        .unwrap_or(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn centroid_symmetric() {
        let samples: Vec<(f64, f64)> = vec![
            (0.0, 0.0), (2.0, 0.5), (4.0, 1.0), (6.0, 0.5), (8.0, 0.0),
        ];
        let c = centroid(&samples);
        assert!((c - 4.0).abs() < 1e-6);
    }

    #[test]
    fn centroid_single_peak() {
        let samples: Vec<(f64, f64)> = vec![
            (0.0, 0.0), (5.0, 1.0), (10.0, 0.0),
        ];
        let c = centroid(&samples);
        assert!((c - 5.0).abs() < 1e-6);
    }

    #[test]
    fn mean_of_maxima_single() {
        let samples: Vec<(f64, f64)> = vec![
            (0.0, 0.0), (2.0, 0.5), (4.0, 1.0), (6.0, 0.5), (8.0, 0.0),
        ];
        let m = mean_of_maxima(&samples);
        assert!((m - 4.0).abs() < 1e-9);
    }

    #[test]
    fn mean_of_maxima_plateau() {
        let samples: Vec<(f64, f64)> = vec![
            (0.0, 0.0), (3.0, 1.0), (5.0, 1.0), (7.0, 1.0), (10.0, 0.0),
        ];
        let m = mean_of_maxima(&samples);
        assert!((m - 5.0).abs() < 1e-9);
    }

    #[test]
    fn bisector_symmetric() {
        let samples: Vec<(f64, f64)> = vec![
            (0.0, 0.0), (2.0, 0.5), (4.0, 1.0), (6.0, 0.5), (8.0, 0.0),
        ];
        let b = bisector(&samples);
        assert!((b - 4.0).abs() < 1e-6);
    }

    #[test]
    fn weighted_average_test() {
        let items = vec![(10.0, 0.3), (20.0, 0.7)];
        let wa = weighted_average(&items);
        assert!((wa - 17.0).abs() < 1e-9);
    }

    #[test]
    fn largest_of_max_test() {
        let samples: Vec<(f64, f64)> = vec![
            (0.0, 0.0), (3.0, 1.0), (5.0, 1.0), (7.0, 0.5), (10.0, 0.0),
        ];
        assert!((largest_of_max(&samples) - 5.0).abs() < 1e-9);
    }

    #[test]
    fn smallest_of_max_test() {
        let samples: Vec<(f64, f64)> = vec![
            (0.0, 0.0), (3.0, 1.0), (5.0, 1.0), (7.0, 0.5), (10.0, 0.0),
        ];
        assert!((smallest_of_max(&samples) - 3.0).abs() < 1e-9);
    }

    #[test]
    fn centroid_empty() {
        let samples: Vec<(f64, f64)> = vec![];
        assert!((centroid(&samples)).abs() < 1e-9);
    }
}
