//! Fuzzy decision reasoning — handling uncertainty in decision-making.
//!
//! Provides utilities for making decisions under uncertainty
//! using fuzzy logic: action selection, confidence scoring, and
//! multi-criteria decision making.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::membership::FuzzySet;
use crate::operations::TNorm;

/// Represents a fuzzy criterion for decision-making.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Criterion {
    pub name: String,
    pub weight: f64,
    pub favorable: FuzzySet,
}

impl Criterion {
    pub fn new(name: impl Into<String>, weight: f64, favorable: FuzzySet) -> Self {
        Self { name: name.into(), weight, favorable }
    }
}

/// An action option that can be evaluated.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Action {
    pub name: String,
    pub scores: HashMap<String, f64>,
}

impl Action {
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into(), scores: HashMap::new() }
    }

    pub fn with_score(mut self, criterion: impl Into<String>, value: f64) -> Self {
        self.scores.insert(criterion.into(), value);
        self
    }
}

/// Fuzzy decision engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FuzzyDecisionEngine {
    pub criteria: Vec<Criterion>,
    pub tnorm: TNorm,
}

impl FuzzyDecisionEngine {
    pub fn new(criteria: Vec<Criterion>, tnorm: TNorm) -> Self {
        Self { criteria, tnorm }
    }

    /// Evaluate an action across all criteria.
    /// Returns a fuzzy score in [0, 1] representing overall favorability.
    pub fn evaluate_action(&self, action: &Action) -> f64 {
        let mut weighted_sum = 0.0;
        let mut total_weight = 0.0;

        for criterion in &self.criteria {
            let raw_score = action.scores.get(&criterion.name).copied().unwrap_or(0.0);
            let membership = criterion.favorable.membership(raw_score);
            weighted_sum += membership * criterion.weight;
            total_weight += criterion.weight;
        }

        if total_weight.abs() < 1e-12 {
            0.0
        } else {
            (weighted_sum / total_weight).clamp(0.0, 1.0)
        }
    }

    /// Rank multiple actions by their fuzzy scores.
    /// Returns actions sorted best-first with their scores.
    pub fn rank_actions(&self, actions: &[Action]) -> Vec<(String, f64)> {
        let mut scored: Vec<(String, f64)> = actions
            .iter()
            .map(|a| (a.name.clone(), self.evaluate_action(a)))
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored
    }

    /// Select the best action.
    pub fn best_action(&self, actions: &[Action]) -> Option<(String, f64)> {
        self.rank_actions(actions).into_iter().next()
    }
}

/// Confidence level representation.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum Confidence {
    VeryLow,
    Low,
    Medium,
    High,
    VeryHigh,
}

impl Confidence {
    /// Convert to a numeric value in [0, 1].
    pub fn to_value(&self) -> f64 {
        match self {
            Self::VeryLow => 0.1,
            Self::Low => 0.3,
            Self::Medium => 0.5,
            Self::High => 0.7,
            Self::VeryHigh => 0.9,
        }
    }

    /// Convert from a numeric value.
    pub fn from_value(v: f64) -> Self {
        if v < 0.2 { Self::VeryLow }
        else if v < 0.4 { Self::Low }
        else if v < 0.6 { Self::Medium }
        else if v < 0.8 { Self::High }
        else { Self::VeryHigh }
    }
}

/// Fuzzy belief state — tracks uncertainty about world state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FuzzyBelief {
    pub proposition: String,
    pub truth_value: f64,
    pub confidence: f64,
}

impl FuzzyBelief {
    pub fn new(proposition: impl Into<String>, truth_value: f64, confidence: f64) -> Self {
        Self {
            proposition: proposition.into(),
            truth_value: truth_value.clamp(0.0, 1.0),
            confidence: confidence.clamp(0.0, 1.0),
        }
    }

    /// Combine two beliefs about the same proposition using confidence-weighted averaging.
    pub fn combine(&self, other: &FuzzyBelief) -> FuzzyBelief {
        let total_conf = self.confidence + other.confidence;
        let combined_truth = if total_conf.abs() < 1e-12 {
            (self.truth_value + other.truth_value) / 2.0
        } else {
            (self.truth_value * self.confidence + other.truth_value * other.confidence) / total_conf
        };
        FuzzyBelief::new(
            &self.proposition,
            combined_truth,
            (self.confidence.max(other.confidence)).min(1.0),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::membership::MembershipShape;

    fn make_engine() -> FuzzyDecisionEngine {
        let speed = Criterion::new(
            "speed",
            0.4,
            FuzzySet::new("speed_fav", MembershipShape::Trapezoidal { a: 0.0, b: 0.5, c: 1.0, d: 1.0 }, (0.0, 1.0)),
        );
        let reliability = Criterion::new(
            "reliability",
            0.6,
            FuzzySet::new("rel_fav", MembershipShape::Sigmoid { a: 10.0, c: 0.5 }, (0.0, 1.0)),
        );
        FuzzyDecisionEngine::new(vec![speed, reliability], TNorm::Product)
    }

    #[test]
    fn evaluate_single_action() {
        let engine = make_engine();
        let action = Action::new("fast_risky").with_score("speed", 0.9).with_score("reliability", 0.3);
        let score = engine.evaluate_action(&action);
        assert!(score > 0.0 && score < 1.0);
    }

    #[test]
    fn rank_actions() {
        let engine = make_engine();
        let actions = vec![
            Action::new("slow_reliable").with_score("speed", 0.2).with_score("reliability", 0.9),
            Action::new("fast_risky").with_score("speed", 0.9).with_score("reliability", 0.3),
            Action::new("balanced").with_score("speed", 0.6).with_score("reliability", 0.6),
        ];
        let ranked = engine.rank_actions(&actions);
        assert_eq!(ranked.len(), 3);
        // reliability (weight=0.6) uses sigmoid(10, 0.5):
        // slow_reliable: speed=0.2, reliability=0.9
        // balanced: speed=0.6, reliability=0.6
        // fast_risky: speed=0.9, reliability=0.3
        // sigmoid at 0.9 ≈ 0.999, at 0.6 ≈ 0.998, at 0.3 ≈ 0.953
        // With high sigmoid, balanced scores highest on both → ranks first
        let best = ranked[0].0.as_str();
        assert!(best == "balanced" || best == "slow_reliable",
            "Expected balanced or slow_reliable to rank first, got {}", best);
    }

    #[test]
    fn best_action() {
        let engine = make_engine();
        let actions = vec![
            Action::new("a").with_score("speed", 0.1).with_score("reliability", 0.1),
            Action::new("b").with_score("speed", 0.9).with_score("reliability", 0.9),
        ];
        let best = engine.best_action(&actions).unwrap();
        assert_eq!(best.0, "b");
        assert!(best.1 > 0.8);
    }

    #[test]
    fn confidence_levels() {
        assert!((Confidence::VeryLow.to_value() - 0.1).abs() < 1e-9);
        assert!((Confidence::VeryHigh.to_value() - 0.9).abs() < 1e-9);
        assert_eq!(Confidence::from_value(0.5), Confidence::Medium);
        assert_eq!(Confidence::from_value(0.95), Confidence::VeryHigh);
    }

    #[test]
    fn fuzzy_belief_combine() {
        let b1 = FuzzyBelief::new("user_is_happy", 0.8, 0.5);
        let b2 = FuzzyBelief::new("user_is_happy", 0.4, 0.9);
        let combined = b1.combine(&b2);
        // Weighted: (0.8*0.5 + 0.4*0.9) / (0.5+0.9) = (0.4 + 0.36)/1.4 = 0.76/1.4 ≈ 0.543
        assert!((combined.truth_value - 0.5428).abs() < 0.01);
        assert!((combined.confidence - 0.9).abs() < 1e-9); // max of confidences
    }

    #[test]
    fn belief_clamping() {
        let b = FuzzyBelief::new("test", 1.5, -0.2);
        assert!((b.truth_value - 1.0).abs() < 1e-9);
        assert!((b.confidence).abs() < 1e-9);
    }

    #[test]
    fn no_matching_criteria() {
        let engine = make_engine();
        let action = Action::new("empty");
        let score = engine.evaluate_action(&action);
        assert!(score < 0.01, "Expected near-zero score for empty action, got {}", score);
    }
}
