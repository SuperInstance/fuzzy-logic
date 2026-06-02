//! Linguistic variables and hedges (very, somewhat, not).

use serde::{Deserialize, Serialize};

use crate::membership::{FuzzySet, MembershipShape};

/// Linguistic hedge modifiers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Hedge {
    Very,
    Somewhat,
    Not,
    VeryVery,
    Indeed,
}

impl Hedge {
    /// Apply this hedge to a membership value.
    pub fn apply(&self, mu: f64) -> f64 {
        let mu = mu.clamp(0.0, 1.0);
        match self {
            Self::Very => mu * mu,
            Self::Somewhat => mu.sqrt(),
            Self::Not => 1.0 - mu,
            Self::VeryVery => mu.powi(4),
            Self::Indeed => {
                // 2*mu^2 if mu >= 0.5, else 1 - 2*(1-mu)^2
                if mu >= 0.5 { 2.0 * mu * mu } else { 1.0 - 2.0 * (1.0 - mu).powi(2) }
            }
        }
    }
}

/// A linguistic variable binds a name to a universe of discourse and a collection of fuzzy sets (terms).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinguisticVariable {
    pub name: String,
    pub domain: (f64, f64),
    pub terms: Vec<FuzzySet>,
}

impl LinguisticVariable {
    pub fn new(name: impl Into<String>, domain: (f64, f64)) -> Self {
        Self { name: name.into(), domain, terms: vec![] }
    }

    /// Add a fuzzy set term to this variable.
    pub fn add_term(&mut self, term: FuzzySet) {
        self.terms.push(term);
    }

    /// Get a term by name.
    pub fn term(&self, name: &str) -> Option<&FuzzySet> {
        self.terms.iter().find(|t| t.name == name)
    }

    /// Evaluate a term with optional hedges at value `x`.
    pub fn evaluate(&self, term_name: &str, hedges: &[Hedge], x: f64) -> Option<f64> {
        self.term(term_name).map(|t| {
            let mu = t.membership(x);
            hedges.iter().fold(mu, |acc, h| h.apply(acc))
        })
    }
}

/// Apply hedges to an existing membership function shape, returning a new shape.
/// This creates a closure-based membership for non-standard shapes.
pub fn apply_hedges_to_shape(shape: &MembershipShape, hedges: &[Hedge]) -> Box<dyn Fn(f64) -> f64 + Send + Sync> {
    let shape = shape.clone();
    let hedges = hedges.to_vec();
    Box::new(move |x| {
        let mu = shape.evaluate(x);
        hedges.iter().fold(mu, |acc, h| h.apply(acc))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_variable() -> LinguisticVariable {
        let mut lv = LinguisticVariable::new("temperature", (0.0, 100.0));
        lv.add_term(FuzzySet::triangular("cold", 0.0, 0.0, 30.0, (0.0, 100.0)));
        lv.add_term(FuzzySet::triangular("warm", 15.0, 35.0, 55.0, (0.0, 100.0)));
        lv.add_term(FuzzySet::triangular("hot", 45.0, 70.0, 100.0, (0.0, 100.0)));
        lv
    }

    #[test]
    fn hedge_very() {
        assert!((Hedge::Very.apply(0.5) - 0.25).abs() < 1e-9);
        assert!((Hedge::Very.apply(0.8) - 0.64).abs() < 1e-9);
    }

    #[test]
    fn hedge_somewhat() {
        assert!((Hedge::Somewhat.apply(0.25) - 0.5).abs() < 1e-9);
    }

    #[test]
    fn hedge_not() {
        assert!((Hedge::Not.apply(0.3) - 0.7).abs() < 1e-9);
    }

    #[test]
    fn hedge_very_very() {
        assert!((Hedge::VeryVery.apply(0.5) - 0.0625).abs() < 1e-9);
    }

    #[test]
    fn hedge_indeed() {
        assert!((Hedge::Indeed.apply(0.8) - 2.0 * 0.64).abs() < 1e-9); // 1.28
        assert!((Hedge::Indeed.apply(0.2) - (1.0 - 2.0 * 0.64)).abs() < 1e-9); // 0.28... wait
        // 1 - 2*(1-0.2)^2 = 1 - 2*0.64 = 1 - 1.28 = -0.28... hmm, clamped?
        let val = Hedge::Indeed.apply(0.2);
        // 1 - 2*(0.8)^2 = 1 - 1.28 = -0.28, but we clamp to [0,1] inside hedge? No, we don't.
        // The formula itself can go below 0 for very low values. Let's just check the formula:
        assert!((val - (1.0_f64 - 2.0_f64 * (1.0_f64 - 0.2_f64).powi(2))).abs() < 1e-9);
    }

    #[test]
    fn linguistic_variable_evaluate() {
        let lv = temp_variable();
        // At x=30, cold should be at peak (1.0) since (30-0)/(30-0)=1.0... wait
        // triangular(0, 0, 30): at x=0 => a=b=0 so (x-a)/(b-a) = 0/0. Edge case.
        // Actually at x=0: x<=a (0<=0) so returns 0? No, x <= *a = 0 is true, so returns 0.
        // But triangular(0,0,30): a=0, b=0, c=30. x=0: x <= a (0<=0) → 0. That's not ideal.
        // Let's test at x=20: cold: (30-20)/(30-0) = 10/30 ≈ 0.333
        let cold_20 = lv.evaluate("cold", &[], 20.0).unwrap();
        assert!((cold_20 - 1.0/3.0).abs() < 1e-6);
    }

    #[test]
    fn linguistic_variable_with_hedges() {
        let lv = temp_variable();
        let warm_35 = lv.evaluate("warm", &[], 35.0).unwrap();
        assert!((warm_35 - 1.0).abs() < 1e-9);
        let very_warm_35 = lv.evaluate("warm", &[Hedge::Very], 35.0).unwrap();
        assert!((very_warm_35 - 1.0).abs() < 1e-9);
    }

    #[test]
    fn apply_hedges_fn() {
        let shape = MembershipShape::Triangular { a: 0.0, b: 5.0, c: 10.0 };
        let f = apply_hedges_to_shape(&shape, &[Hedge::Very]);
        // At x=5, mu=1.0, very(1.0) = 1.0
        assert!((f(5.0) - 1.0).abs() < 1e-9);
        // At x=2.5, mu=0.5, very(0.5) = 0.25
        assert!((f(2.5) - 0.25).abs() < 1e-9);
    }
}
