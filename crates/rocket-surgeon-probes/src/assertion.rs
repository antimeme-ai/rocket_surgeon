use winnow::ModalResult;
use winnow::ascii::{alpha1, digit1, space0};
use winnow::combinator::{alt, opt, preceded, repeat};
use winnow::prelude::*;
use winnow::token::one_of;

use rocket_surgeon_protocol::types::TensorStats;

#[derive(Debug, Clone, PartialEq)]
pub struct Assertion {
    pub field: StatsField,
    pub op: CmpOp,
    pub value: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StatsField {
    Mean,
    Std,
    Min,
    Max,
    AbsMax,
    Sparsity,
    L2Norm,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CmpOp {
    Lt,
    Gt,
    Le,
    Ge,
    Eq,
    Ne,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("invalid assertion: {message}")]
pub struct AssertionParseError {
    pub message: String,
}

impl Assertion {
    pub fn parse(input: &str) -> Result<Self, AssertionParseError> {
        winnow::Parser::parse(&mut assertion, input).map_err(|e| AssertionParseError {
            message: e.to_string(),
        })
    }

    pub fn evaluate(&self, stats: &TensorStats) -> bool {
        let lhs = self.field.extract(stats);
        match self.op {
            CmpOp::Lt => lhs < self.value,
            CmpOp::Gt => lhs > self.value,
            CmpOp::Le => lhs <= self.value,
            CmpOp::Ge => lhs >= self.value,
            CmpOp::Eq => (lhs - self.value).abs() < f64::EPSILON,
            CmpOp::Ne => (lhs - self.value).abs() >= f64::EPSILON,
        }
    }
}

impl StatsField {
    fn extract(self, stats: &TensorStats) -> f64 {
        match self {
            Self::Mean => stats.mean,
            Self::Std => stats.std,
            Self::Min => stats.min,
            Self::Max => stats.max,
            Self::AbsMax => stats.abs_max,
            Self::Sparsity => stats.sparsity,
            Self::L2Norm => stats.l2_norm,
        }
    }
}

// --- Parser ---

fn field_name(input: &mut &str) -> ModalResult<StatsField> {
    let name: String = {
        let first: &str = alpha1.parse_next(input)?;
        let rest: Vec<char> = repeat(0.., one_of(|c: char| c.is_ascii_alphanumeric() || c == '_'))
            .parse_next(input)?;
        let mut s = first.to_string();
        s.extend(rest);
        s
    };
    match name.as_str() {
        "mean" => Ok(StatsField::Mean),
        "std" => Ok(StatsField::Std),
        "min" => Ok(StatsField::Min),
        "max" => Ok(StatsField::Max),
        "abs_max" => Ok(StatsField::AbsMax),
        "sparsity" => Ok(StatsField::Sparsity),
        "l2_norm" | "norm" => Ok(StatsField::L2Norm),
        _ => Err(winnow::error::ErrMode::Cut(
            winnow::error::ContextError::new(),
        )),
    }
}

fn cmp_op(input: &mut &str) -> ModalResult<CmpOp> {
    alt((
        "<=".map(|_| CmpOp::Le),
        ">=".map(|_| CmpOp::Ge),
        "!=".map(|_| CmpOp::Ne),
        "==".map(|_| CmpOp::Eq),
        "<".map(|_| CmpOp::Lt),
        ">".map(|_| CmpOp::Gt),
    ))
    .parse_next(input)
}

fn float_literal(input: &mut &str) -> ModalResult<f64> {
    let neg: Option<char> = opt('-').parse_next(input)?;
    let int_part: &str = digit1.parse_next(input)?;
    let frac: Option<(char, &str)> = opt(('.', digit1)).parse_next(input)?;
    let exp: Option<(char, Option<char>, &str)> =
        opt((one_of(['e', 'E']), opt(one_of(['+', '-'])), digit1)).parse_next(input)?;

    let mut s = String::new();
    if neg.is_some() {
        s.push('-');
    }
    s.push_str(int_part);
    if let Some((_, frac_digits)) = frac {
        s.push('.');
        s.push_str(frac_digits);
    }
    if let Some((e, sign, exp_digits)) = exp {
        s.push(e);
        if let Some(sign_char) = sign {
            s.push(sign_char);
        }
        s.push_str(exp_digits);
    }

    s.parse::<f64>()
        .map_err(|_| winnow::error::ErrMode::Cut(winnow::error::ContextError::new()))
}

fn assertion(input: &mut &str) -> ModalResult<Assertion> {
    let field = preceded(space0, field_name).parse_next(input)?;
    let op = preceded(space0, cmp_op).parse_next(input)?;
    let value = preceded(space0, float_literal).parse_next(input)?;
    Ok(Assertion { field, op, value })
}

#[cfg(test)]
mod tests {
    use super::*;
    use rocket_surgeon_protocol::types::Histogram;

    fn sample_stats() -> TensorStats {
        TensorStats {
            mean: 2.5,
            std: 1.0,
            min: 0.0,
            max: 5.0,
            abs_max: 5.0,
            sparsity: 0.1,
            l2_norm: 10.0,
            histogram: Histogram {
                bins: 10,
                edges: vec![],
                counts: vec![],
            },
        }
    }

    #[test]
    fn parse_norm_lt() {
        let a = Assertion::parse("norm < 100.0").unwrap();
        assert_eq!(a.field, StatsField::L2Norm);
        assert_eq!(a.op, CmpOp::Lt);
        assert!((a.value - 100.0).abs() < f64::EPSILON);
    }

    #[test]
    fn parse_sparsity_gt() {
        let a = Assertion::parse("sparsity > 0.5").unwrap();
        assert_eq!(a.field, StatsField::Sparsity);
        assert_eq!(a.op, CmpOp::Gt);
    }

    #[test]
    fn parse_abs_max_le() {
        let a = Assertion::parse("abs_max <= 1e6").unwrap();
        assert_eq!(a.field, StatsField::AbsMax);
        assert_eq!(a.op, CmpOp::Le);
        assert!((a.value - 1e6).abs() < 1.0);
    }

    #[test]
    fn parse_mean_eq() {
        let a = Assertion::parse("mean == 0.0").unwrap();
        assert_eq!(a.field, StatsField::Mean);
        assert_eq!(a.op, CmpOp::Eq);
    }

    #[test]
    fn parse_std_ne() {
        let a = Assertion::parse("std != 0.0").unwrap();
        assert_eq!(a.field, StatsField::Std);
        assert_eq!(a.op, CmpOp::Ne);
    }

    #[test]
    fn parse_l2_norm_alias() {
        let a = Assertion::parse("l2_norm < 50.0").unwrap();
        assert_eq!(a.field, StatsField::L2Norm);
    }

    #[test]
    fn evaluate_norm_lt_passes() {
        let a = Assertion::parse("norm < 100.0").unwrap();
        assert!(a.evaluate(&sample_stats()));
    }

    #[test]
    fn evaluate_norm_lt_fails() {
        let a = Assertion::parse("norm < 5.0").unwrap();
        assert!(!a.evaluate(&sample_stats()));
    }

    #[test]
    fn evaluate_sparsity_gt_fails() {
        let a = Assertion::parse("sparsity > 0.5").unwrap();
        assert!(!a.evaluate(&sample_stats()));
    }

    #[test]
    fn evaluate_max_ge() {
        let a = Assertion::parse("max >= 5.0").unwrap();
        assert!(a.evaluate(&sample_stats()));
    }

    #[test]
    fn parse_invalid_field_rejects() {
        assert!(Assertion::parse("bogus < 1.0").is_err());
    }

    #[test]
    fn parse_missing_operator_rejects() {
        assert!(Assertion::parse("mean 1.0").is_err());
    }

    #[test]
    fn parse_negative_literal() {
        let a = Assertion::parse("min > -10.0").unwrap();
        assert_eq!(a.op, CmpOp::Gt);
        assert!((a.value - (-10.0)).abs() < f64::EPSILON);
    }

    #[test]
    fn parse_scientific_notation() {
        let a = Assertion::parse("abs_max < 1e3").unwrap();
        assert!((a.value - 1000.0).abs() < f64::EPSILON);
    }
}
