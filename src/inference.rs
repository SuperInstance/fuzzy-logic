//! Fuzzy inference: Mamdani and Sugeno.

use serde::{Deserialize, Serialize};

use crate::operations::TNorm;
use crate::rules::{Condition, RuleBase};
use crate::linguistic::LinguisticVariable;

/// Result of a fuzzy inference step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceResult {
    /// Firing strengths per rule: (rule_index, strength)
    pub firing_strengths: Vec<(usize, f64)>,
    /// Aggregated output as sampled points (for Mamdani)
    pub aggregated: Vec<(f64, f64)>,
    /// Individual rule outputs (for Sugeno)
    pub rule_outputs: Vec<f64>,
}

/// Mamdani fuzzy inference system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MamdaniInference {
    pub input_variables: Vec<LinguisticVariable>,
    pub output_variable: LinguisticVariable,
    pub rule_base: RuleBase,
    pub tnorm: TNorm,
}

impl MamdaniInference {
    pub fn new(
        input_variables: Vec<LinguisticVariable>,
        output_variable: LinguisticVariable,
        rule_base: RuleBase,
        tnorm: TNorm,
    ) -> Self {
        Self { input_variables, output_variable, rule_base, tnorm }
    }

    /// Run inference for given input values.
    /// `inputs` maps variable names to crisp values.
    pub fn infer(&self, inputs: &std::collections::HashMap<String, f64>, sample_count: usize) -> InferenceResult {
        let firing = self.rule_base.fire_all(
            |condition| {
                self.compute_antecedent_membership(condition, inputs)
            },
            &self.tnorm,
        );

        // For each fired rule, clip/implicate the output fuzzy set
        let domain = self.output_variable.domain;
        let step = (domain.1 - domain.0) / sample_count as f64;

        let mut aggregated: Vec<(f64, f64)> = Vec::with_capacity(sample_count + 1);
        for i in 0..=sample_count {
            let x = domain.0 + step * i as f64;
            let mut max_mu: f64 = 0.0;
            for (rule_idx, strength) in &firing {
                let rule = &self.rule_base.rules[*rule_idx];
                let output_term = self.output_variable.term(&rule.consequent_term);
                if let Some(term) = output_term {
                    let mu = term.membership(x);
                    // Mamdani implication: min(strength, mu)
                    let implicated = self.tnorm.apply(*strength, mu);
                    max_mu = max_mu.max(implicated);
                }
            }
            aggregated.push((x, max_mu));
        }

        InferenceResult {
            firing_strengths: firing,
            aggregated,
            rule_outputs: vec![],
        }
    }

    fn compute_antecedent_membership(&self, condition: &Condition, inputs: &std::collections::HashMap<String, f64>) -> f64 {
        let x = match inputs.get(&condition.variable) {
            Some(v) => *v,
            None => return 0.0,
        };

        for var in &self.input_variables {
            if var.name == condition.variable {
                return var.evaluate(&condition.term, &[], x).unwrap_or(0.0);
            }
        }
        0.0
    }
}

/// Sugeno output function for a rule.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SugenoOutput {
    /// Linear: z = a1*x1 + a2*x2 + ... + b
    Linear { coefficients: Vec<f64>, constant: f64 },
    /// Constant output.
    Constant(f64),
}

impl SugenoOutput {
    /// Evaluate with given input values (ordered same as coefficients).
    pub fn evaluate(&self, inputs: &[f64]) -> f64 {
        match self {
            Self::Constant(c) => *c,
            Self::Linear { coefficients, constant } => {
                coefficients.iter().zip(inputs.iter()).map(|(a, x)| a * x).sum::<f64>() + constant
            }
        }
    }
}

/// Sugeno fuzzy inference system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SugenoInference {
    pub input_variables: Vec<LinguisticVariable>,
    pub rule_base: RuleBase,
    pub outputs: Vec<SugenoOutput>,
    pub tnorm: TNorm,
}

impl SugenoInference {
    pub fn new(
        input_variables: Vec<LinguisticVariable>,
        rule_base: RuleBase,
        outputs: Vec<SugenoOutput>,
        tnorm: TNorm,
    ) -> Self {
        Self { input_variables, rule_base, outputs, tnorm }
    }

    /// Run Sugeno inference. Returns weighted average output.
    /// `inputs` maps variable names to crisp values, in order matching coefficients.
    pub fn infer(&self, inputs: &std::collections::HashMap<String, f64>, input_order: &[String]) -> InferenceResult {
        let input_vals: Vec<f64> = input_order.iter().map(|name| *inputs.get(name).unwrap_or(&0.0)).collect();

        let firing = self.rule_base.fire_all(
            |condition| {
                let x = *inputs.get(&condition.variable).unwrap_or(&0.0);
                for var in &self.input_variables {
                    if var.name == condition.variable {
                        return var.evaluate(&condition.term, &[], x).unwrap_or(0.0);
                    }
                }
                0.0
            },
            &self.tnorm,
        );

        let mut weighted_sum = 0.0;
        let mut total_weight = 0.0;
        let mut rule_outputs = Vec::new();

        for (rule_idx, strength) in &firing {
            let output = self.outputs.get(*rule_idx);
            let z = output.map(|o| o.evaluate(&input_vals)).unwrap_or(0.0);
            rule_outputs.push(z);
            weighted_sum += strength * z;
            total_weight += strength;
        }

        let crisp = if total_weight.abs() > 1e-12 {
            weighted_sum / total_weight
        } else {
            0.0
        };

        // Store aggregated as single point for Sugeno
        InferenceResult {
            firing_strengths: firing,
            aggregated: vec![(crisp, 1.0)],
            rule_outputs,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{FuzzySet, FuzzyRule};
    use std::collections::HashMap;

    fn make_mamdani() -> MamdaniInference {
        let mut temp = LinguisticVariable::new("temperature", (0.0, 100.0));
        temp.add_term(FuzzySet::triangular("cold", 0.0, 20.0, 40.0, (0.0, 100.0)));
        temp.add_term(FuzzySet::triangular("hot", 40.0, 70.0, 100.0, (0.0, 100.0)));

        let mut fan = LinguisticVariable::new("fan_speed", (0.0, 100.0));
        fan.add_term(FuzzySet::triangular("slow", 0.0, 20.0, 40.0, (0.0, 100.0)));
        fan.add_term(FuzzySet::triangular("fast", 50.0, 80.0, 100.0, (0.0, 100.0)));

        let rules = RuleBase::new(vec![
            FuzzyRule::new(vec![Condition::new("temperature", "cold")], "fan_speed", "slow"),
            FuzzyRule::new(vec![Condition::new("temperature", "hot")], "fan_speed", "fast"),
        ]);

        MamdaniInference::new(vec![temp], fan, rules, TNorm::Minimum)
    }

    #[test]
    fn mamdani_hot_input() {
        let sys = make_mamdani();
        let mut inputs = HashMap::new();
        inputs.insert("temperature".into(), 70.0);
        let result = sys.infer(&inputs, 100);
        // At 70, cold = 0, hot = 1.0 → rule 1 fires 0, rule 2 fires 1.0
        assert!((result.firing_strengths[0].1).abs() < 1e-9); // cold fires at 0
        assert!((result.firing_strengths[1].1 - 1.0).abs() < 1e-9); // hot fires at 1
    }

    #[test]
    fn mamdani_overlap() {
        let sys = make_mamdani();
        let mut inputs = HashMap::new();
        inputs.insert("temperature".into(), 40.0);
        let result = sys.infer(&inputs, 100);
        // At 40: cold peak is 20, so cold(40) = (40-40)/(40-0) wait...
        // triangular cold(0,20,40): at x=40, (40-20)/(40-20)... actually:
        // x=40 >= c=40 → 0. No wait: c=40, so x>=c → 0.
        // cold(40) = 0. hot(40): triangular(40,70,100): a=40, at x=40, x<=a → 0.
        // So both fire at 0 at the boundary.
        assert!(result.firing_strengths[0].1 < 1e-9);
        assert!(result.firing_strengths[1].1 < 1e-9);
    }

    #[test]
    fn sugeno_inference() {
        let mut temp = LinguisticVariable::new("temperature", (0.0, 100.0));
        temp.add_term(FuzzySet::triangular("cold", 0.0, 20.0, 40.0, (0.0, 100.0)));
        temp.add_term(FuzzySet::triangular("hot", 40.0, 70.0, 100.0, (0.0, 100.0)));

        let rules = RuleBase::new(vec![
            FuzzyRule::new(vec![Condition::new("temperature", "cold")], "output", "low"),
            FuzzyRule::new(vec![Condition::new("temperature", "hot")], "output", "high"),
        ]);

        let outputs = vec![
            SugenoOutput::Constant(10.0), // cold → 10
            SugenoOutput::Constant(90.0), // hot → 90
        ];

        let sys = SugenoInference::new(vec![temp], rules, outputs, TNorm::Minimum);

        let mut inputs = HashMap::new();
        inputs.insert("temperature".into(), 70.0);
        let result = sys.infer(&inputs, &["temperature".into()]);
        // cold(70) = 0, hot(70) = 1.0 → weighted = 90/1 = 90
        assert!((result.aggregated[0].0 - 90.0).abs() < 1e-6);
    }

    #[test]
    fn sugeno_linear_output() {
        let mut x_var = LinguisticVariable::new("x", (0.0, 10.0));
        x_var.add_term(FuzzySet::triangular("low", 0.0, 2.5, 5.0, (0.0, 10.0)));
        x_var.add_term(FuzzySet::triangular("high", 5.0, 7.5, 10.0, (0.0, 10.0)));

        let rules = RuleBase::new(vec![
            FuzzyRule::new(vec![Condition::new("x", "low")], "y", "rule1"),
            FuzzyRule::new(vec![Condition::new("x", "high")], "y", "rule2"),
        ]);

        let outputs = vec![
            SugenoOutput::Linear { coefficients: vec![2.0], constant: 1.0 },
            SugenoOutput::Linear { coefficients: vec![0.5], constant: 5.0 },
        ];

        let sys = SugenoInference::new(vec![x_var], rules, outputs, TNorm::Product);
        let mut inputs = HashMap::new();
        inputs.insert("x".into(), 2.5);
        let result = sys.infer(&inputs, &["x".into()]);
        // low(2.5) = 1.0, high(2.5) = 0.0
        // z1 = 2*2.5 + 1 = 6, z2 = 0.5*2.5 + 5 = 6.25
        // output = (1.0*6 + 0.0*6.25) / 1.0 = 6.0
        assert!((result.aggregated[0].0 - 6.0).abs() < 1e-6);
    }

    #[test]
    fn mamdani_result_has_samples() {
        let sys = make_mamdani();
        let mut inputs = HashMap::new();
        inputs.insert("temperature".into(), 50.0);
        let result = sys.infer(&inputs, 50);
        assert_eq!(result.aggregated.len(), 51);
    }
}
