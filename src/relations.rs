//! Fuzzy relations and composition using nalgebra.

use nalgebra::DMatrix;
use serde::{Deserialize, Serialize};

use crate::operations::TNorm;

/// A fuzzy relation represented as a matrix of membership values.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FuzzyRelation {
    pub matrix: DMatrix<f64>,
    pub row_labels: Vec<String>,
    pub col_labels: Vec<String>,
}

impl FuzzyRelation {
    pub fn new(row_labels: Vec<String>, col_labels: Vec<String>, data: Vec<f64>) -> Self {
        let rows = row_labels.len();
        let cols = col_labels.len();
        let matrix = DMatrix::from_row_iterator(rows, cols, data.into_iter());
        Self { matrix, row_labels, col_labels }
    }

    /// Get membership value for row i, col j.
    pub fn get(&self, i: usize, j: usize) -> f64 {
        self.matrix[(i, j)]
    }

    /// Max-min composition of two fuzzy relations.
    /// self (n×m) composed with other (m×p) → result (n×p).
    pub fn compose_max_min(&self, other: &FuzzyRelation) -> FuzzyRelation {
        let n = self.matrix.nrows();
        let m = self.matrix.ncols();
        let p = other.matrix.ncols();
        assert_eq!(m, other.matrix.nrows(), "Inner dimensions must match for composition");

        let mut result = vec![0.0; n * p];
        for i in 0..n {
            for j in 0..p {
                let mut max_val: f64 = 0.0;
                for k in 0..m {
                    let min_val = self.matrix[(i, k)].min(other.matrix[(k, j)]);
                    max_val = max_val.max(min_val);
                }
                result[i * p + j] = max_val;
            }
        }

        let row_labels = self.row_labels.clone();
        let col_labels = other.col_labels.clone();
        FuzzyRelation::new(row_labels, col_labels, result)
    }

    /// Composition using a general t-norm instead of min.
    pub fn compose_tnorm(&self, other: &FuzzyRelation, tnorm: &TNorm) -> FuzzyRelation {
        let n = self.matrix.nrows();
        let m = self.matrix.ncols();
        let p = other.matrix.ncols();
        assert_eq!(m, other.matrix.nrows());

        let mut result = vec![0.0; n * p];
        for i in 0..n {
            for j in 0..p {
                let mut max_val: f64 = 0.0;
                for k in 0..m {
                    let val = tnorm.apply(self.matrix[(i, k)], other.matrix[(k, j)]);
                    max_val = max_val.max(val);
                }
                result[i * p + j] = max_val;
            }
        }

        FuzzyRelation::new(self.row_labels.clone(), other.col_labels.clone(), result)
    }

    /// Transpose the fuzzy relation.
    pub fn transpose(&self) -> FuzzyRelation {
        FuzzyRelation {
            matrix: self.matrix.transpose(),
            row_labels: self.col_labels.clone(),
            col_labels: self.row_labels.clone(),
        }
    }

    /// Check if the relation is reflexive (square, diagonal all 1.0).
    pub fn is_reflexive(&self, tolerance: f64) -> bool {
        let n = self.matrix.nrows();
        if n != self.matrix.ncols() { return false; }
        for i in 0..n {
            if (self.matrix[(i, i)] - 1.0).abs() > tolerance {
                return false;
            }
        }
        true
    }

    /// Check if the relation is symmetric.
    pub fn is_symmetric(&self, tolerance: f64) -> bool {
        let n = self.matrix.nrows();
        if n != self.matrix.ncols() { return false; }
        for i in 0..n {
            for j in 0..n {
                if (self.matrix[(i, j)] - self.matrix[(j, i)]).abs() > tolerance {
                    return false;
                }
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn relation_creation() {
        let r = FuzzyRelation::new(
            vec!["a".into(), "b".into()],
            vec!["x".into(), "y".into()],
            vec![0.1, 0.8, 0.6, 0.3],
        );
        assert!((r.get(0, 0) - 0.1).abs() < 1e-9);
        assert!((r.get(0, 1) - 0.8).abs() < 1e-9);
        assert!((r.get(1, 0) - 0.6).abs() < 1e-9);
        assert!((r.get(1, 1) - 0.3).abs() < 1e-9);
    }

    #[test]
    fn max_min_composition() {
        // R1: 2×3, R2: 3×2
        let r1 = FuzzyRelation::new(
            vec!["a".into(), "b".into()],
            vec!["x".into(), "y".into(), "z".into()],
            vec![0.1, 0.5, 0.3, 0.8, 0.2, 0.7],
        );
        let r2 = FuzzyRelation::new(
            vec!["x".into(), "y".into(), "z".into()],
            vec!["p".into(), "q".into()],
            vec![0.9, 0.1, 0.4, 0.6, 0.2, 0.8],
        );
        let result = r1.compose_max_min(&r2);
        // a,p: max(min(0.1,0.9), min(0.5,0.4), min(0.3,0.2)) = max(0.1, 0.4, 0.2) = 0.4
        assert!((result.get(0, 0) - 0.4).abs() < 1e-9);
        // a,q: max(min(0.1,0.1), min(0.5,0.6), min(0.3,0.8)) = max(0.1, 0.5, 0.3) = 0.5
        assert!((result.get(0, 1) - 0.5).abs() < 1e-9);
        assert_eq!(result.row_labels.len(), 2);
        assert_eq!(result.col_labels.len(), 2);
    }

    #[test]
    fn reflexivity() {
        let r = FuzzyRelation::new(
            vec!["a".into(), "b".into()],
            vec!["a".into(), "b".into()],
            vec![1.0, 0.3, 0.5, 1.0],
        );
        assert!(r.is_reflexive(1e-9));
    }

    #[test]
    fn symmetry() {
        let r = FuzzyRelation::new(
            vec!["a".into(), "b".into()],
            vec!["a".into(), "b".into()],
            vec![1.0, 0.5, 0.5, 1.0],
        );
        assert!(r.is_symmetric(1e-9));
    }

    #[test]
    fn transpose() {
        let r = FuzzyRelation::new(
            vec!["a".into(), "b".into()],
            vec!["x".into(), "y".into()],
            vec![0.1, 0.8, 0.6, 0.3],
        );
        let t = r.transpose();
        assert!((t.get(0, 0) - 0.1).abs() < 1e-9);
        assert!((t.get(0, 1) - 0.6).abs() < 1e-9);
        assert!((t.get(1, 0) - 0.8).abs() < 1e-9);
    }
}
