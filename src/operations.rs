//! Fuzzy set operations: union, intersection, complement, t-norms, t-conorms.

use serde::{Deserialize, Serialize};

/// T-norm (fuzzy intersection) variants.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum TNorm {
    Minimum,
    Product,
    Lukasiewicz { p: f64 },
    Drastic,
    Hamacher { gamma: f64 },
}

impl TNorm {
    pub fn apply(&self, a: f64, b: f64) -> f64 {
        let a = a.clamp(0.0, 1.0);
        let b = b.clamp(0.0, 1.0);
        match self {
            Self::Minimum => a.min(b),
            Self::Product => a * b,
            Self::Lukasiewicz { .. } => (a + b - 1.0).max(0.0),
            Self::Drastic => {
                if a == 1.0 { b } else if b == 1.0 { a } else { 0.0 }
            }
            Self::Hamacher { gamma } => {
                let denom = gamma + (1.0 - gamma) * (a + b - a * b);
                if denom.abs() < 1e-12 { 0.0 } else { a * b / denom }
            }
        }
    }
}

/// T-conorm (fuzzy union) variants.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum TConorm {
    Maximum,
    ProbabilisticSum,
    BoundedSum,
    Drastic,
    EinsteinSum,
}

impl TConorm {
    pub fn apply(&self, a: f64, b: f64) -> f64 {
        let a = a.clamp(0.0, 1.0);
        let b = b.clamp(0.0, 1.0);
        match self {
            Self::Maximum => a.max(b),
            Self::ProbabilisticSum => a + b - a * b,
            Self::BoundedSum => (a + b).min(1.0),
            Self::Drastic => {
                if a == 0.0 { b } else if b == 0.0 { a } else { 1.0 }
            }
            Self::EinsteinSum => (a + b) / (1.0 + a * b),
        }
    }
}

/// Standard fuzzy complement.
pub fn complement(a: f64) -> f64 {
    1.0 - a.clamp(0.0, 1.0)
}

/// Yager complement with parameter w.
pub fn yager_complement(a: f64, w: f64) -> f64 {
    (1.0 - a.clamp(0.0, 1.0).powf(w)).powf(1.0 / w)
}

/// Sugeno complement with parameter lambda.
pub fn sugeno_complement(a: f64, lambda: f64) -> f64 {
    (1.0 - a.clamp(0.0, 1.0)) / (1.0 + lambda * a.clamp(0.0, 1.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tnorm_minimum() {
        assert!((TNorm::Minimum.apply(0.6, 0.4) - 0.4).abs() < 1e-9);
    }

    #[test]
    fn tnorm_product() {
        assert!((TNorm::Product.apply(0.5, 0.8) - 0.4).abs() < 1e-9);
    }

    #[test]
    fn tnorm_lukasiewicz() {
        assert!((TNorm::Lukasiewicz { p: 1.0 }.apply(0.7, 0.6) - 0.3).abs() < 1e-9);
        assert!((TNorm::Lukasiewicz { p: 1.0 }.apply(0.2, 0.3)).abs() < 1e-9);
    }

    #[test]
    fn tnorm_drastic() {
        assert!((TNorm::Drastic.apply(1.0, 0.5) - 0.5).abs() < 1e-9);
        assert!((TNorm::Drastic.apply(0.8, 0.7)).abs() < 1e-9);
    }

    #[test]
    fn tconorm_maximum() {
        assert!((TConorm::Maximum.apply(0.6, 0.4) - 0.6).abs() < 1e-9);
    }

    #[test]
    fn tconorm_probabilistic() {
        assert!((TConorm::ProbabilisticSum.apply(0.5, 0.5) - 0.75).abs() < 1e-9);
    }

    #[test]
    fn tconorm_bounded() {
        assert!((TConorm::BoundedSum.apply(0.7, 0.6) - 1.0).abs() < 1e-9);
        assert!((TConorm::BoundedSum.apply(0.3, 0.4) - 0.7).abs() < 1e-9);
    }

    #[test]
    fn complement_standard() {
        assert!((complement(0.3) - 0.7).abs() < 1e-9);
        assert!((complement(0.0) - 1.0).abs() < 1e-9);
        assert!((complement(1.0)).abs() < 1e-9);
    }

    #[test]
    fn yager_complement_test() {
        let c = yager_complement(0.5, 2.0);
        assert!((c - (1.0 - 0.25_f64).sqrt()).abs() < 1e-6);
    }

    #[test]
    fn sugeno_complement_test() {
        let c = sugeno_complement(0.5, 0.0);
        assert!((c - 0.5).abs() < 1e-9);
    }

    #[test]
    fn de_morgan_tnorm_tconorm() {
        let a = 0.6;
        let b = 0.4;
        // complement(min(a,b)) == max(complement(a), complement(b))
        let left = complement(TNorm::Minimum.apply(a, b));
        let right = TConorm::Maximum.apply(complement(a), complement(b));
        assert!((left - right).abs() < 1e-9);
    }
}
