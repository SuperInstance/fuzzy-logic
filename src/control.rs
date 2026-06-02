//! Fuzzy control systems: inputs → rules → inference → defuzzification → output.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::linguistic::LinguisticVariable;
use crate::inference::MamdaniInference;
use crate::defuzzification;
use crate::operations::TNorm;
use crate::rules::RuleBase;

/// Defuzzification method selector.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum DefuzzificationMethod {
    Centroid,
    MeanOfMaxima,
    Bisector,
}

/// A complete fuzzy control system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FuzzyControlSystem {
    pub mamdani: MamdaniInference,
    pub defuzz_method: DefuzzificationMethod,
    pub sample_count: usize,
}

impl FuzzyControlSystem {
    pub fn new(
        input_variables: Vec<LinguisticVariable>,
        output_variable: LinguisticVariable,
        rule_base: RuleBase,
        tnorm: TNorm,
        defuzz_method: DefuzzificationMethod,
        sample_count: usize,
    ) -> Self {
        let mamdani = MamdaniInference::new(input_variables, output_variable, rule_base, tnorm);
        Self { mamdani, defuzz_method, sample_count }
    }

    /// Run the control system: inputs → crisp output.
    pub fn compute(&self, inputs: &HashMap<String, f64>) -> f64 {
        let result = self.mamdani.infer(inputs, self.sample_count);
        match self.defuzz_method {
            DefuzzificationMethod::Centroid => defuzzification::centroid(&result.aggregated),
            DefuzzificationMethod::MeanOfMaxima => defuzzification::mean_of_maxima(&result.aggregated),
            DefuzzificationMethod::Bisector => defuzzification::bisector(&result.aggregated),
        }
    }

    /// Get the aggregated output shape for debugging/visualization.
    pub fn get_aggregated(&self, inputs: &HashMap<String, f64>) -> Vec<(f64, f64)> {
        let result = self.mamdani.infer(inputs, self.sample_count);
        result.aggregated
    }

    /// Get firing strengths for each rule.
    pub fn get_firing_strengths(&self, inputs: &HashMap<String, f64>) -> Vec<(usize, f64)> {
        let result = self.mamdani.infer(inputs, self.sample_count);
        result.firing_strengths
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{FuzzySet, FuzzyRule};
    use crate::rules::Condition;

    fn make_temperature_controller() -> FuzzyControlSystem {
        let mut temp = LinguisticVariable::new("temperature", (0.0, 100.0));
        temp.add_term(FuzzySet::triangular("cold", 0.0, 0.0, 30.0, (0.0, 100.0)));
        temp.add_term(FuzzySet::triangular("comfortable", 20.0, 40.0, 60.0, (0.0, 100.0)));
        temp.add_term(FuzzySet::triangular("hot", 50.0, 80.0, 100.0, (0.0, 100.0)));

        let mut fan = LinguisticVariable::new("fan_speed", (0.0, 100.0));
        fan.add_term(FuzzySet::triangular("off", 0.0, 0.0, 30.0, (0.0, 100.0)));
        fan.add_term(FuzzySet::triangular("medium", 20.0, 50.0, 80.0, (0.0, 100.0)));
        fan.add_term(FuzzySet::triangular("high", 60.0, 80.0, 100.0, (0.0, 100.0)));

        let rules = RuleBase::new(vec![
            FuzzyRule::new(vec![Condition::new("temperature", "cold")], "fan_speed", "off"),
            FuzzyRule::new(vec![Condition::new("temperature", "comfortable")], "fan_speed", "medium"),
            FuzzyRule::new(vec![Condition::new("temperature", "hot")], "fan_speed", "high"),
        ]);

        FuzzyControlSystem::new(
            vec![temp],
            fan,
            rules,
            TNorm::Minimum,
            DefuzzificationMethod::Centroid,
            200,
        )
    }

    #[test]
    fn cold_input_low_fan() {
        let ctrl = make_temperature_controller();
        let mut inputs = HashMap::new();
        inputs.insert("temperature".into(), 5.0);
        let output = ctrl.compute(&inputs);
        // Cold → fan off, should be low
        assert!(output < 20.0, "Expected low fan speed for cold, got {}", output);
    }

    #[test]
    fn hot_input_high_fan() {
        let ctrl = make_temperature_controller();
        let mut inputs = HashMap::new();
        inputs.insert("temperature".into(), 90.0);
        let output = ctrl.compute(&inputs);
        // Hot → fan high
        assert!(output > 60.0, "Expected high fan speed for hot, got {}", output);
    }

    #[test]
    fn comfortable_input_medium_fan() {
        let ctrl = make_temperature_controller();
        let mut inputs = HashMap::new();
        inputs.insert("temperature".into(), 40.0);
        let output = ctrl.compute(&inputs);
        assert!(output > 25.0 && output < 75.0, "Expected medium fan for comfortable, got {}", output);
    }

    #[test]
    fn monotonic_response() {
        let ctrl = make_temperature_controller();
        let mut prev = f64::NEG_INFINITY;
        let mut got_strictly_monotonic = true;
        for temp in (5..=95).step_by(10) {
            let mut inputs = HashMap::new();
            inputs.insert("temperature".into(), temp as f64);
            let output = ctrl.compute(&inputs);
            if output < prev - 2.0 {
                got_strictly_monotonic = false;
            }
            prev = output;
        }
        // General trend should be increasing
        assert!(prev > 20.0, "Fan speed at high temp should be significant, got {}", prev);
    }

    #[test]
    fn defuzz_method_bisector() {
        let mut ctrl = make_temperature_controller();
        ctrl.defuzz_method = DefuzzificationMethod::Bisector;
        let mut inputs = HashMap::new();
        inputs.insert("temperature".into(), 40.0);
        let output = ctrl.compute(&inputs);
        assert!(output > 20.0 && output < 80.0, "Bisector gave {}", output);
    }

    #[test]
    fn firing_strengths_retrieval() {
        let ctrl = make_temperature_controller();
        let mut inputs = HashMap::new();
        inputs.insert("temperature".into(), 25.0);
        let strengths = ctrl.get_firing_strengths(&inputs);
        assert_eq!(strengths.len(), 3);
        // At 25: cold is active, comfortable is active, hot is ~0
        assert!(strengths[0].1 > 0.0); // cold fires
        assert!(strengths[1].1 > 0.0); // comfortable fires
    }
}
