# fuzzy-logic

**Fuzzy logic in Rust. When true/false isn't enough.**

Membership functions (triangular, trapezoidal, Gaussian, sigmoid), t-norms/t-conorms, linguistic variables with hedges, IF-THEN rule bases, Mamdani and Sugeno inference, defuzzification, fuzzy control systems, and fuzzy decision-making.

## Install

```toml
[dependencies]
fuzzy-logic = "0.1"
```

## Quick Start

```rust
use fuzzy_logic::*;
use std::collections::HashMap;

// Membership functions
let cold = FuzzySet::triangular("cold", 0.0, 0.0, 30.0, (0.0, 100.0));
let warm = FuzzySet::triangular("warm", 15.0, 35.0, 55.0, (0.0, 100.0));
let hot  = FuzzySet::triangular("hot",  45.0, 70.0, 100.0, (0.0, 100.0));

println!("cold(10) = {}", cold.membership(10.0)); // ~0.67

// Fuzzy control system
let mut temperature = LinguisticVariable::new("temperature", (0.0, 100.0));
temperature.add_term(cold);
temperature.add_term(FuzzySet::triangular("comfortable", 20.0, 40.0, 60.0, (0.0, 100.0)));
temperature.add_term(hot);

let mut fan_speed = LinguisticVariable::new("fan_speed", (0.0, 100.0));
fan_speed.add_term(FuzzySet::triangular("off", 0.0, 0.0, 30.0, (0.0, 100.0)));
fan_speed.add_term(FuzzySet::triangular("medium", 20.0, 50.0, 80.0, (0.0, 100.0)));
fan_speed.add_term(FuzzySet::triangular("high", 60.0, 80.0, 100.0, (0.0, 100.0)));

let rules = RuleBase::new(vec![
    FuzzyRule::new(vec![Condition::new("temperature", "cold")], "fan_speed", "off"),
    FuzzyRule::new(vec![Condition::new("temperature", "comfortable")], "fan_speed", "medium"),
    FuzzyRule::new(vec![Condition::new("temperature", "hot")], "fan_speed", "high"),
]);

let ctrl = FuzzyControlSystem::new(
    vec![temperature], fan_speed, rules,
    TNorm::Minimum, DefuzzificationMethod::Centroid, 200,
);

let mut inputs = HashMap::new();
inputs.insert("temperature".into(), 25.0);
println!("Fan speed at 25°C: {}", ctrl.compute(&inputs));
```

## Testing

```sh
cargo test
```

## License

MIT OR Apache-2.0
