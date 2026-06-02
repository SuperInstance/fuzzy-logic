//! # fuzzy-logic
//!
//! Fuzzy logic in Rust — fuzzy sets, membership functions, defuzzification, fuzzy control systems.

pub mod membership;
pub mod operations;
pub mod relations;
pub mod linguistic;
pub mod rules;
pub mod inference;
pub mod defuzzification;
pub mod control;
pub mod decision;

pub use membership::*;
pub use operations::*;
pub use relations::*;
pub use linguistic::*;
pub use rules::*;
pub use inference::*;
pub use defuzzification::*;
pub use control::*;
