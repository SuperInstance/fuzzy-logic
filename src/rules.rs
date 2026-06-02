//! Fuzzy IF-THEN rules and rule firing.

use serde::{Deserialize, Serialize};

use crate::operations::TNorm;

/// A condition in a fuzzy rule's antecedent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Condition {
    pub variable: String,
    pub term: String,
}

impl Condition {
    pub fn new(variable: impl Into<String>, term: impl Into<String>) -> Self {
        Self { variable: variable.into(), term: term.into() }
    }
}

/// A fuzzy IF-THEN rule.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FuzzyRule {
    pub antecedents: Vec<Condition>,
    pub consequent_variable: String,
    pub consequent_term: String,
    pub weight: f64,
}

impl FuzzyRule {
    pub fn new(
        antecedents: Vec<Condition>,
        consequent_variable: impl Into<String>,
        consequent_term: impl Into<String>,
    ) -> Self {
        Self {
            antecedents,
            consequent_variable: consequent_variable.into(),
            consequent_term: consequent_term.into(),
            weight: 1.0,
        }
    }

    pub fn with_weight(mut self, w: f64) -> Self {
        self.weight = w;
        self
    }

    /// Fire this rule given membership values for each antecedent condition.
    /// Uses the specified t-norm to combine antecedents.
    /// Returns the firing strength.
    pub fn fire(&self, memberships: &[f64], tnorm: &TNorm) -> f64 {
        if memberships.is_empty() {
            return 0.0;
        }
        let strength = memberships.iter().copied().fold(1.0, |acc, mu| tnorm.apply(acc, mu));
        strength * self.weight
    }
}

/// A fuzzy rule base.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuleBase {
    pub rules: Vec<FuzzyRule>,
}

impl RuleBase {
    pub fn new(rules: Vec<FuzzyRule>) -> Self {
        Self { rules }
    }

    /// Fire all rules and return (rule_index, firing_strength) pairs.
    pub fn fire_all<F>(&self, membership_fn: F, tnorm: &TNorm) -> Vec<(usize, f64)>
    where
        F: Fn(&Condition) -> f64,
    {
        self.rules
            .iter()
            .enumerate()
            .map(|(i, rule)| {
                let mus: Vec<f64> = rule.antecedents.iter().map(|c| membership_fn(c)).collect();
                (i, rule.fire(&mus, tnorm))
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_antecedent_rule() {
        let rule = FuzzyRule::new(
            vec![Condition::new("temp", "hot")],
            "fan",
            "fast",
        );
        let strength = rule.fire(&[0.8], &TNorm::Minimum);
        assert!((strength - 0.8).abs() < 1e-9);
    }

    #[test]
    fn multi_antecedent_rule_min() {
        let rule = FuzzyRule::new(
            vec![Condition::new("temp", "hot"), Condition::new("humidity", "high")],
            "fan",
            "fast",
        );
        let strength = rule.fire(&[0.7, 0.5], &TNorm::Minimum);
        assert!((strength - 0.5).abs() < 1e-9);
    }

    #[test]
    fn multi_antecedent_rule_product() {
        let rule = FuzzyRule::new(
            vec![Condition::new("temp", "hot"), Condition::new("humidity", "high")],
            "fan",
            "fast",
        );
        let strength = rule.fire(&[0.7, 0.5], &TNorm::Product);
        assert!((strength - 0.35).abs() < 1e-9);
    }

    #[test]
    fn rule_with_weight() {
        let rule = FuzzyRule::new(
            vec![Condition::new("temp", "hot")],
            "fan",
            "fast",
        ).with_weight(0.5);
        let strength = rule.fire(&[0.8], &TNorm::Minimum);
        assert!((strength - 0.4).abs() < 1e-9);
    }

    #[test]
    fn rule_base_fire_all() {
        let rb = RuleBase::new(vec![
            FuzzyRule::new(vec![Condition::new("temp", "cold")], "fan", "off"),
            FuzzyRule::new(vec![Condition::new("temp", "warm")], "fan", "medium"),
            FuzzyRule::new(vec![Condition::new("temp", "hot")], "fan", "fast"),
        ]);
        let results = rb.fire_all(
            |c| match c.term.as_str() {
                "cold" => 0.2,
                "warm" => 0.8,
                "hot" => 0.3,
                _ => 0.0,
            },
            &TNorm::Minimum,
        );
        assert_eq!(results.len(), 3);
        assert!((results[0].1 - 0.2).abs() < 1e-9);
        assert!((results[1].1 - 0.8).abs() < 1e-9);
        assert!((results[2].1 - 0.3).abs() < 1e-9);
    }

    #[test]
    fn empty_antecedents() {
        let rule = FuzzyRule::new(vec![], "out", "term");
        let strength = rule.fire(&[], &TNorm::Minimum);
        assert!((strength).abs() < 1e-9);
    }
}
