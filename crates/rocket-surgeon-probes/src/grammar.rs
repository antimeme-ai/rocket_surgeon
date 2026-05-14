use std::fmt;

use serde::{Deserialize, Serialize};
use winnow::ModalResult;
use winnow::ascii::{alpha1, digit1};
use winnow::combinator::{alt, opt, preceded, repeat, separated, terminated};
use winnow::prelude::*;
use winnow::token::one_of;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProbePoint {
    pub model: NameOrWild,
    pub rank: NumOrWild,
    pub layer: NumOrWild,
    pub component: ComponentOrWild,
    pub event: NameOrWild,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum NameOrWild {
    Wildcard,
    Name(String),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum NumOrWild {
    Wildcard,
    Num(u32),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ComponentOrWild {
    Wildcard,
    Path(Vec<ComponentSeg>),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ComponentSeg {
    Named(String),
    Indexed { name: String, index: u32 },
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("invalid probe point: {message}")]
pub struct ParseError {
    pub message: String,
    pub offset: usize,
}

impl ProbePoint {
    pub fn parse(input: &str) -> Result<Self, ParseError> {
        winnow::Parser::parse(&mut probe_point, input).map_err(|e| ParseError {
            message: e.to_string(),
            offset: e.offset(),
        })
    }

    pub fn matches(&self, other: &Self) -> bool {
        name_matches(&self.model, &other.model)
            && num_matches(&self.rank, &other.rank)
            && num_matches(&self.layer, &other.layer)
            && component_matches(&self.component, &other.component)
            && name_matches(&self.event, &other.event)
    }
}

fn name_matches(pattern: &NameOrWild, target: &NameOrWild) -> bool {
    match (pattern, target) {
        (NameOrWild::Wildcard, _) | (_, NameOrWild::Wildcard) => true,
        (NameOrWild::Name(a), NameOrWild::Name(b)) => a == b,
    }
}

fn num_matches(pattern: &NumOrWild, target: &NumOrWild) -> bool {
    match (pattern, target) {
        (NumOrWild::Wildcard, _) | (_, NumOrWild::Wildcard) => true,
        (NumOrWild::Num(a), NumOrWild::Num(b)) => a == b,
    }
}

fn component_matches(pattern: &ComponentOrWild, target: &ComponentOrWild) -> bool {
    match (pattern, target) {
        (ComponentOrWild::Wildcard, _) | (_, ComponentOrWild::Wildcard) => true,
        (ComponentOrWild::Path(a), ComponentOrWild::Path(b)) => a == b,
    }
}

impl fmt::Display for ProbePoint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}:{}:{}:{}:{}",
            self.model, self.rank, self.layer, self.component, self.event
        )
    }
}

impl fmt::Display for NameOrWild {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Wildcard => f.write_str("*"),
            Self::Name(n) => f.write_str(n),
        }
    }
}

impl fmt::Display for NumOrWild {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Wildcard => f.write_str("*"),
            Self::Num(n) => write!(f, "{n}"),
        }
    }
}

impl fmt::Display for ComponentOrWild {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Wildcard => f.write_str("*"),
            Self::Path(segs) => {
                for (i, seg) in segs.iter().enumerate() {
                    if i > 0 {
                        f.write_str(".")?;
                    }
                    write!(f, "{seg}")?;
                }
                Ok(())
            }
        }
    }
}

impl fmt::Display for ComponentSeg {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Named(n) => f.write_str(n),
            Self::Indexed { name, index } => write!(f, "{name}[{index}]"),
        }
    }
}

// --- Parser combinators ---

fn identifier(input: &mut &str) -> ModalResult<String> {
    let first: &str = alpha1.parse_next(input)?;
    let rest: Vec<char> = repeat(
        0..,
        one_of(|c: char| c.is_ascii_alphanumeric() || c == '_' || c == '-'),
    )
    .parse_next(input)?;
    let mut result = first.to_string();
    result.extend(rest);
    Ok(result)
}

fn non_neg_integer(input: &mut &str) -> ModalResult<u32> {
    let digits: &str = digit1.parse_next(input)?;
    digits
        .parse::<u32>()
        .map_err(|_| winnow::error::ErrMode::Cut(winnow::error::ContextError::new()))
}

fn wildcard(input: &mut &str) -> ModalResult<()> {
    '*'.void().parse_next(input)
}

fn name_or_wild(input: &mut &str) -> ModalResult<NameOrWild> {
    alt((
        wildcard.map(|()| NameOrWild::Wildcard),
        identifier.map(NameOrWild::Name),
    ))
    .parse_next(input)
}

fn num_or_wild(input: &mut &str) -> ModalResult<NumOrWild> {
    alt((
        wildcard.map(|()| NumOrWild::Wildcard),
        non_neg_integer.map(NumOrWild::Num),
    ))
    .parse_next(input)
}

fn component_seg(input: &mut &str) -> ModalResult<ComponentSeg> {
    let name = identifier.parse_next(input)?;
    let idx: Option<u32> =
        opt(preceded('[', terminated(non_neg_integer, ']'))).parse_next(input)?;
    Ok(match idx {
        Some(index) => ComponentSeg::Indexed { name, index },
        None => ComponentSeg::Named(name),
    })
}

fn component_or_wild(input: &mut &str) -> ModalResult<ComponentOrWild> {
    alt((
        wildcard.map(|()| ComponentOrWild::Wildcard),
        separated(1.., component_seg, '.').map(ComponentOrWild::Path),
    ))
    .parse_next(input)
}

fn probe_point(input: &mut &str) -> ModalResult<ProbePoint> {
    let model = name_or_wild.parse_next(input)?;
    ':'.void().parse_next(input)?;
    let rank = num_or_wild.parse_next(input)?;
    ':'.void().parse_next(input)?;
    let layer = num_or_wild.parse_next(input)?;
    ':'.void().parse_next(input)?;
    let component = component_or_wild.parse_next(input)?;
    ':'.void().parse_next(input)?;
    let event = name_or_wild.parse_next(input)?;
    Ok(ProbePoint {
        model,
        rank,
        layer,
        component,
        event,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- Design doc §8 examples ---

    #[test]
    fn parse_attn_output_all_ranks() {
        let p = ProbePoint::parse("llama:*:12:attn.o_proj:output").unwrap();
        assert_eq!(p.model, NameOrWild::Name("llama".into()));
        assert_eq!(p.rank, NumOrWild::Wildcard);
        assert_eq!(p.layer, NumOrWild::Num(12));
        assert_eq!(
            p.component,
            ComponentOrWild::Path(vec![
                ComponentSeg::Named("attn".into()),
                ComponentSeg::Named("o_proj".into()),
            ])
        );
        assert_eq!(p.event, NameOrWild::Name("output".into()));
    }

    #[test]
    fn parse_mlp_input_rank0() {
        let p = ProbePoint::parse("llama:0:*:mlp:input").unwrap();
        assert_eq!(p.model, NameOrWild::Name("llama".into()));
        assert_eq!(p.rank, NumOrWild::Num(0));
        assert_eq!(p.layer, NumOrWild::Wildcard);
        assert_eq!(
            p.component,
            ComponentOrWild::Path(vec![ComponentSeg::Named("mlp".into())])
        );
        assert_eq!(p.event, NameOrWild::Name("input".into()));
    }

    #[test]
    fn parse_moe_router_pre_topk() {
        let p = ProbePoint::parse("mixtral:*:8:router:pre_topk").unwrap();
        assert_eq!(p.model, NameOrWild::Name("mixtral".into()));
        assert_eq!(p.rank, NumOrWild::Wildcard);
        assert_eq!(p.layer, NumOrWild::Num(8));
        assert_eq!(
            p.component,
            ComponentOrWild::Path(vec![ComponentSeg::Named("router".into())])
        );
        assert_eq!(p.event, NameOrWild::Name("pre_topk".into()));
    }

    #[test]
    fn parse_all_wildcards() {
        let p = ProbePoint::parse("llama:*:*:residual_post:*").unwrap();
        assert_eq!(p.model, NameOrWild::Name("llama".into()));
        assert_eq!(p.rank, NumOrWild::Wildcard);
        assert_eq!(p.layer, NumOrWild::Wildcard);
        assert_eq!(
            p.component,
            ComponentOrWild::Path(vec![ComponentSeg::Named("residual_post".into())])
        );
        assert_eq!(p.event, NameOrWild::Wildcard);
    }

    #[test]
    fn parse_attn_scores_virtual() {
        let p = ProbePoint::parse("llama:0:12:attn.scores:*").unwrap();
        assert_eq!(p.rank, NumOrWild::Num(0));
        assert_eq!(p.layer, NumOrWild::Num(12));
        assert_eq!(
            p.component,
            ComponentOrWild::Path(vec![
                ComponentSeg::Named("attn".into()),
                ComponentSeg::Named("scores".into()),
            ])
        );
    }

    #[test]
    fn parse_indexed_expert() {
        let p = ProbePoint::parse("mixtral:*:8:experts[3]:output").unwrap();
        assert_eq!(
            p.component,
            ComponentOrWild::Path(vec![ComponentSeg::Indexed {
                name: "experts".into(),
                index: 3,
            }])
        );
    }

    #[test]
    fn parse_indexed_expert_with_subcomponent() {
        let p = ProbePoint::parse("mixtral:*:8:experts[3].gate_proj:output").unwrap();
        assert_eq!(
            p.component,
            ComponentOrWild::Path(vec![
                ComponentSeg::Indexed {
                    name: "experts".into(),
                    index: 3,
                },
                ComponentSeg::Named("gate_proj".into()),
            ])
        );
    }

    #[test]
    fn parse_wildcard_component() {
        let p = ProbePoint::parse("llama:*:12:*:output").unwrap();
        assert_eq!(p.component, ComponentOrWild::Wildcard);
    }

    #[test]
    fn parse_full_wildcard() {
        let p = ProbePoint::parse("*:*:*:*:*").unwrap();
        assert_eq!(p.model, NameOrWild::Wildcard);
        assert_eq!(p.rank, NumOrWild::Wildcard);
        assert_eq!(p.layer, NumOrWild::Wildcard);
        assert_eq!(p.component, ComponentOrWild::Wildcard);
        assert_eq!(p.event, NameOrWild::Wildcard);
    }

    // --- Round-trip ---

    #[test]
    fn round_trip_complex() {
        let input = "mixtral:*:8:experts[3].gate_proj:output";
        let parsed = ProbePoint::parse(input).unwrap();
        assert_eq!(parsed.to_string(), input);
    }

    #[test]
    fn round_trip_all_wildcards() {
        let input = "*:*:*:*:*";
        let parsed = ProbePoint::parse(input).unwrap();
        assert_eq!(parsed.to_string(), input);
    }

    #[test]
    fn round_trip_all_concrete() {
        let input = "llama:0:12:attn.o_proj:output";
        let parsed = ProbePoint::parse(input).unwrap();
        assert_eq!(parsed.to_string(), input);
    }

    // --- Wildcard matching ---

    #[test]
    fn wildcard_matches_any_rank() {
        let pattern = ProbePoint::parse("llama:*:12:mlp:output").unwrap();
        let target = ProbePoint::parse("llama:3:12:mlp:output").unwrap();
        assert!(pattern.matches(&target));
    }

    #[test]
    fn wildcard_matches_any_layer() {
        let pattern = ProbePoint::parse("llama:0:*:mlp:output").unwrap();
        let target = ProbePoint::parse("llama:0:7:mlp:output").unwrap();
        assert!(pattern.matches(&target));
    }

    #[test]
    fn wildcard_matches_any_component() {
        let pattern = ProbePoint::parse("llama:0:12:*:output").unwrap();
        let target = ProbePoint::parse("llama:0:12:attn.o_proj:output").unwrap();
        assert!(pattern.matches(&target));
    }

    #[test]
    fn concrete_does_not_match_different() {
        let pattern = ProbePoint::parse("llama:0:12:mlp:output").unwrap();
        let target = ProbePoint::parse("llama:0:12:attn:output").unwrap();
        assert!(!pattern.matches(&target));
    }

    #[test]
    fn different_model_does_not_match() {
        let pattern = ProbePoint::parse("llama:0:12:mlp:output").unwrap();
        let target = ProbePoint::parse("mixtral:0:12:mlp:output").unwrap();
        assert!(!pattern.matches(&target));
    }

    #[test]
    fn different_layer_does_not_match() {
        let pattern = ProbePoint::parse("llama:0:12:mlp:output").unwrap();
        let target = ProbePoint::parse("llama:0:13:mlp:output").unwrap();
        assert!(!pattern.matches(&target));
    }

    // --- Invalid input ---

    #[test]
    fn reject_missing_segments() {
        assert!(ProbePoint::parse("llama:0:12:mlp").is_err());
    }

    #[test]
    fn reject_empty_string() {
        assert!(ProbePoint::parse("").is_err());
    }

    #[test]
    fn reject_extra_segment() {
        assert!(ProbePoint::parse("llama:0:12:mlp:output:extra").is_err());
    }

    #[test]
    fn reject_trailing_colon() {
        assert!(ProbePoint::parse("llama:0:12:mlp:output:").is_err());
    }

    #[test]
    fn reject_leading_colon() {
        assert!(ProbePoint::parse(":0:12:mlp:output").is_err());
    }

    #[test]
    fn reject_negative_layer() {
        assert!(ProbePoint::parse("llama:0:-1:mlp:output").is_err());
    }

    #[test]
    fn reject_non_numeric_rank() {
        assert!(ProbePoint::parse("llama:abc:12:mlp:output").is_err());
    }

    #[test]
    fn reject_empty_component_segment() {
        assert!(ProbePoint::parse("llama:0:12:.mlp:output").is_err());
    }

    #[test]
    fn reject_unclosed_bracket() {
        assert!(ProbePoint::parse("llama:0:12:experts[3:output").is_err());
    }

    // --- Serde round-trip ---

    #[test]
    fn serde_round_trip() {
        let p = ProbePoint::parse("llama:0:12:attn.o_proj:output").unwrap();
        let json = serde_json::to_string(&p).unwrap();
        let p2: ProbePoint = serde_json::from_str(&json).unwrap();
        assert_eq!(p, p2);
    }
}
