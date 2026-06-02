//! Fuzzy set membership functions.

use serde::{Deserialize, Serialize};

/// A membership function maps a crisp value to a degree of membership in [0, 1].
pub type MembershipFn = Box<dyn Fn(f64) -> f64 + Send + Sync>;

/// Predefined membership function shapes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MembershipShape {
    Triangular { a: f64, b: f64, c: f64 },
    Trapezoidal { a: f64, b: f64, c: f64, d: f64 },
    Gaussian { mean: f64, sigma: f64 },
    Sigmoid { a: f64, c: f64 },
}

impl MembershipShape {
    /// Evaluate membership at `x`.
    pub fn evaluate(&self, x: f64) -> f64 {
        match self {
            Self::Triangular { a, b, c } => {
                if x <= *a || x >= *c {
                    0.0
                } else if x <= *b {
                    (x - a) / (b - a)
                } else {
                    (c - x) / (c - b)
                }
            }
            Self::Trapezoidal { a, b, c, d } => {
                if x <= *a || x >= *d {
                    0.0
                } else if x < *b {
                    (x - a) / (b - a)
                } else if x <= *c {
                    1.0
                } else {
                    (d - x) / (d - c)
                }
            }
            Self::Gaussian { mean, sigma } => {
                (-((x - mean).powi(2)) / (2.0 * sigma.powi(2))).exp()
            }
            Self::Sigmoid { a, c } => 1.0 / (1.0 + (-a * (x - c)).exp()),
        }
    }
}

/// A fuzzy set defined by a membership function shape over a named domain.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FuzzySet {
    pub name: String,
    pub shape: MembershipShape,
    pub domain: (f64, f64),
}

impl FuzzySet {
    pub fn new(name: impl Into<String>, shape: MembershipShape, domain: (f64, f64)) -> Self {
        Self { name: name.into(), shape, domain }
    }

    /// Evaluate the membership degree at `x`.
    pub fn membership(&self, x: f64) -> f64 {
        self.shape.evaluate(x)
    }

    /// Sample membership values across the domain at `n` equally spaced points.
    pub fn sample(&self, n: usize) -> Vec<(f64, f64)> {
        let (lo, hi) = self.domain;
        (0..=n)
            .map(|i| {
                let x = lo + (hi - lo) * i as f64 / n as f64;
                (x, self.membership(x))
            })
            .collect()
    }
}

/// Convenience constructors.
impl FuzzySet {
    pub fn triangular(name: impl Into<String>, a: f64, b: f64, c: f64, domain: (f64, f64)) -> Self {
        Self::new(name, MembershipShape::Triangular { a, b, c }, domain)
    }

    pub fn trapezoidal(name: impl Into<String>, a: f64, b: f64, c: f64, d: f64, domain: (f64, f64)) -> Self {
        Self::new(name, MembershipShape::Trapezoidal { a, b, c, d }, domain)
    }

    pub fn gaussian(name: impl Into<String>, mean: f64, sigma: f64, domain: (f64, f64)) -> Self {
        Self::new(name, MembershipShape::Gaussian { mean, sigma }, domain)
    }

    pub fn sigmoid(name: impl Into<String>, a: f64, c: f64, domain: (f64, f64)) -> Self {
        Self::new(name, MembershipShape::Sigmoid { a, c }, domain)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn triangular_peak() {
        let s = FuzzySet::triangular("test", 0.0, 5.0, 10.0, (0.0, 10.0));
        assert!((s.membership(5.0) - 1.0).abs() < 1e-9);
        assert!((s.membership(0.0)).abs() < 1e-9);
        assert!((s.membership(10.0)).abs() < 1e-9);
        assert!((s.membership(2.5) - 0.5).abs() < 1e-9);
        assert!((s.membership(7.5) - 0.5).abs() < 1e-9);
    }

    #[test]
    fn trapezoidal_plateau() {
        let s = FuzzySet::trapezoidal("test", 0.0, 3.0, 7.0, 10.0, (0.0, 10.0));
        assert!((s.membership(5.0) - 1.0).abs() < 1e-9);
        assert!((s.membership(3.0) - 1.0).abs() < 1e-9);
        assert!((s.membership(7.0) - 1.0).abs() < 1e-9);
        assert!((s.membership(1.5) - 0.5).abs() < 1e-9);
        assert!((s.membership(8.5) - 0.5).abs() < 1e-9);
    }

    #[test]
    fn gaussian_bell_curve() {
        let s = FuzzySet::gaussian("test", 5.0, 1.0, (0.0, 10.0));
        assert!((s.membership(5.0) - 1.0).abs() < 1e-9);
        // At mean±sigma, should be e^(-0.5) ≈ 0.6065
        assert!((s.membership(6.0) - (-0.5_f64).exp()).abs() < 1e-6);
    }

    #[test]
    fn sigmoid_s_curve() {
        let s = FuzzySet::sigmoid("test", 2.0, 5.0, (0.0, 10.0));
        // At c, sigmoid = 0.5
        assert!((s.membership(5.0) - 0.5).abs() < 1e-9);
        // Far right → ~1
        assert!(s.membership(10.0) > 0.99);
        // Far left → ~0
        assert!(s.membership(0.0) < 0.01);
    }

    #[test]
    fn sample_count() {
        let s = FuzzySet::triangular("test", 0.0, 5.0, 10.0, (0.0, 10.0));
        let samples = s.sample(10);
        assert_eq!(samples.len(), 11); // 0..=10
    }

    #[test]
    fn membership_bounds() {
        let s = FuzzySet::triangular("test", 0.0, 5.0, 10.0, (0.0, 10.0));
        for x in (-5..=15).map(|i| i as f64) {
            let m = s.membership(x);
            assert!(m >= 0.0 && m <= 1.0, "membership at {} = {}", x, m);
        }
    }
}
