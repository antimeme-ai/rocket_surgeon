#![allow(dead_code)]

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModuleMapping {
    Direct { canonical: String },
    Fused { components: Vec<FusedComponent> },
    Container,
    Skip,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FusedComponent {
    pub canonical: String,
    pub split_dim: i64,
    pub split_size: usize,
}

#[derive(Debug, Clone)]
pub enum ModuleMatcher {
    TypeAndName {
        type_name: &'static str,
        attr_name: &'static str,
    },
    TypeOnly {
        type_name: &'static str,
    },
}

#[derive(Debug, Clone)]
pub struct FamilyDeclaration {
    pub model_types: &'static [&'static str],
    pub mappings: Vec<(ModuleMatcher, ModuleMapping)>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MappedComponent {
    pub module_path: String,
    pub canonical: String,
    pub layer_index: Option<u32>,
    pub call_index: u32,
    pub mapping: ModuleMapping,
    pub probe_point: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ComponentMap {
    pub components: Vec<MappedComponent>,
    pub model_family: String,
    pub vocabulary: Vec<String>,
}

#[allow(clippy::too_many_lines)]
fn llama_declaration() -> FamilyDeclaration {
    use ModuleMapping::{Container, Direct, Skip};
    use ModuleMatcher::{TypeAndName, TypeOnly};
    FamilyDeclaration {
        model_types: &["llama", "mistral", "codellama"],
        mappings: vec![
            (
                TypeOnly {
                    type_name: "LlamaAttention",
                },
                Container,
            ),
            (
                TypeOnly {
                    type_name: "LlamaSdpaAttention",
                },
                Container,
            ),
            (
                TypeOnly {
                    type_name: "LlamaFlashAttention2",
                },
                Container,
            ),
            (
                TypeOnly {
                    type_name: "MistralAttention",
                },
                Container,
            ),
            (
                TypeOnly {
                    type_name: "MistralSdpaAttention",
                },
                Container,
            ),
            (
                TypeOnly {
                    type_name: "MistralFlashAttention2",
                },
                Container,
            ),
            (
                TypeOnly {
                    type_name: "LlamaMLP",
                },
                Container,
            ),
            (
                TypeOnly {
                    type_name: "MistralMLP",
                },
                Container,
            ),
            (
                TypeOnly {
                    type_name: "LlamaDecoderLayer",
                },
                Container,
            ),
            (
                TypeOnly {
                    type_name: "MistralDecoderLayer",
                },
                Container,
            ),
            (
                TypeAndName {
                    type_name: "LlamaRMSNorm",
                    attr_name: "input_layernorm",
                },
                Direct {
                    canonical: "ln1".to_owned(),
                },
            ),
            (
                TypeAndName {
                    type_name: "LlamaRMSNorm",
                    attr_name: "post_attention_layernorm",
                },
                Direct {
                    canonical: "ln2".to_owned(),
                },
            ),
            (
                TypeAndName {
                    type_name: "LlamaRMSNorm",
                    attr_name: "norm",
                },
                Direct {
                    canonical: "ln_final".to_owned(),
                },
            ),
            (
                TypeAndName {
                    type_name: "MistralRMSNorm",
                    attr_name: "input_layernorm",
                },
                Direct {
                    canonical: "ln1".to_owned(),
                },
            ),
            (
                TypeAndName {
                    type_name: "MistralRMSNorm",
                    attr_name: "post_attention_layernorm",
                },
                Direct {
                    canonical: "ln2".to_owned(),
                },
            ),
            (
                TypeAndName {
                    type_name: "MistralRMSNorm",
                    attr_name: "norm",
                },
                Direct {
                    canonical: "ln_final".to_owned(),
                },
            ),
            (
                TypeAndName {
                    type_name: "Linear",
                    attr_name: "q_proj",
                },
                Direct {
                    canonical: "q_proj".to_owned(),
                },
            ),
            (
                TypeAndName {
                    type_name: "Linear",
                    attr_name: "k_proj",
                },
                Direct {
                    canonical: "k_proj".to_owned(),
                },
            ),
            (
                TypeAndName {
                    type_name: "Linear",
                    attr_name: "v_proj",
                },
                Direct {
                    canonical: "v_proj".to_owned(),
                },
            ),
            (
                TypeAndName {
                    type_name: "Linear",
                    attr_name: "o_proj",
                },
                Direct {
                    canonical: "o_proj".to_owned(),
                },
            ),
            (
                TypeAndName {
                    type_name: "Linear",
                    attr_name: "gate_proj",
                },
                Direct {
                    canonical: "gate_proj".to_owned(),
                },
            ),
            (
                TypeAndName {
                    type_name: "Linear",
                    attr_name: "up_proj",
                },
                Direct {
                    canonical: "up_proj".to_owned(),
                },
            ),
            (
                TypeAndName {
                    type_name: "Linear",
                    attr_name: "down_proj",
                },
                Direct {
                    canonical: "down_proj".to_owned(),
                },
            ),
            (
                TypeOnly {
                    type_name: "LlamaRotaryEmbedding",
                },
                Skip,
            ),
            (
                TypeAndName {
                    type_name: "Embedding",
                    attr_name: "embed_tokens",
                },
                Direct {
                    canonical: "embed".to_owned(),
                },
            ),
            (
                TypeAndName {
                    type_name: "Linear",
                    attr_name: "lm_head",
                },
                Direct {
                    canonical: "lm_head".to_owned(),
                },
            ),
        ],
    }
}

#[allow(clippy::too_many_lines)]
fn gpt2_declaration() -> FamilyDeclaration {
    use ModuleMapping::{Container, Direct, Fused, Skip};
    use ModuleMatcher::{TypeAndName, TypeOnly};
    FamilyDeclaration {
        model_types: &["gpt2"],
        mappings: vec![
            (
                TypeOnly {
                    type_name: "GPT2Attention",
                },
                Container,
            ),
            (
                TypeOnly {
                    type_name: "GPT2MLP",
                },
                Container,
            ),
            (
                TypeOnly {
                    type_name: "GPT2Block",
                },
                Container,
            ),
            (
                TypeAndName {
                    type_name: "LayerNorm",
                    attr_name: "ln_1",
                },
                Direct {
                    canonical: "ln1".to_owned(),
                },
            ),
            (
                TypeAndName {
                    type_name: "LayerNorm",
                    attr_name: "ln_2",
                },
                Direct {
                    canonical: "ln2".to_owned(),
                },
            ),
            (
                TypeAndName {
                    type_name: "LayerNorm",
                    attr_name: "ln_f",
                },
                Direct {
                    canonical: "ln_final".to_owned(),
                },
            ),
            (
                TypeAndName {
                    type_name: "Conv1D",
                    attr_name: "c_attn",
                },
                Fused {
                    components: vec![
                        FusedComponent {
                            canonical: "q_proj".to_owned(),
                            split_dim: -1,
                            split_size: 0,
                        },
                        FusedComponent {
                            canonical: "k_proj".to_owned(),
                            split_dim: -1,
                            split_size: 0,
                        },
                        FusedComponent {
                            canonical: "v_proj".to_owned(),
                            split_dim: -1,
                            split_size: 0,
                        },
                    ],
                },
            ),
            (
                TypeAndName {
                    type_name: "Conv1D",
                    attr_name: "c_proj",
                },
                Direct {
                    canonical: "o_proj".to_owned(),
                },
            ),
            (
                TypeAndName {
                    type_name: "Conv1D",
                    attr_name: "c_fc",
                },
                Direct {
                    canonical: "up_proj".to_owned(),
                },
            ),
            (
                TypeAndName {
                    type_name: "Conv1D",
                    attr_name: "c_proj",
                },
                Direct {
                    canonical: "down_proj".to_owned(),
                },
            ),
            (
                TypeAndName {
                    type_name: "Embedding",
                    attr_name: "wte",
                },
                Direct {
                    canonical: "embed".to_owned(),
                },
            ),
            (
                TypeAndName {
                    type_name: "Embedding",
                    attr_name: "wpe",
                },
                Direct {
                    canonical: "pos_embed".to_owned(),
                },
            ),
            (
                TypeAndName {
                    type_name: "Linear",
                    attr_name: "lm_head",
                },
                Direct {
                    canonical: "lm_head".to_owned(),
                },
            ),
            (
                TypeOnly {
                    type_name: "Dropout",
                },
                Skip,
            ),
            (
                TypeOnly {
                    type_name: "NewGELUActivation",
                },
                Skip,
            ),
        ],
    }
}

static FAMILIES: &[fn() -> FamilyDeclaration] = &[llama_declaration, gpt2_declaration];

pub fn family_declaration(model_type: &str) -> Option<FamilyDeclaration> {
    for factory in FAMILIES {
        let decl = factory();
        if decl.model_types.contains(&model_type) {
            return Some(decl);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn direct_mapping_creation() {
        let m = ModuleMapping::Direct {
            canonical: "q_proj".to_owned(),
        };
        assert!(matches!(m, ModuleMapping::Direct { .. }));
    }

    #[test]
    fn fused_mapping_with_three_components() {
        let m = ModuleMapping::Fused {
            components: vec![
                FusedComponent {
                    canonical: "q_proj".to_owned(),
                    split_dim: -1,
                    split_size: 64,
                },
                FusedComponent {
                    canonical: "k_proj".to_owned(),
                    split_dim: -1,
                    split_size: 64,
                },
                FusedComponent {
                    canonical: "v_proj".to_owned(),
                    split_dim: -1,
                    split_size: 64,
                },
            ],
        };
        if let ModuleMapping::Fused { components } = &m {
            assert_eq!(components.len(), 3);
            assert_eq!(components[0].canonical, "q_proj");
        } else {
            panic!("expected Fused");
        }
    }

    #[test]
    fn llama_family_lookup() {
        let decl = family_declaration("llama");
        assert!(decl.is_some());
        let decl = decl.unwrap();
        assert_eq!(decl.model_types, &["llama", "mistral", "codellama"]);
    }

    #[test]
    fn gpt2_family_lookup() {
        let decl = family_declaration("gpt2");
        assert!(decl.is_some());
    }

    #[test]
    fn unknown_family_returns_none() {
        assert!(family_declaration("unknown_arch_xyz").is_none());
    }

    #[test]
    fn llama_has_q_proj_mapping() {
        let decl = family_declaration("llama").unwrap();
        let found = decl.mappings.iter().any(|(matcher, mapping)| {
            matches!(matcher, ModuleMatcher::TypeAndName { attr_name, .. } if *attr_name == "q_proj")
                && matches!(mapping, ModuleMapping::Direct { canonical } if canonical == "q_proj")
        });
        assert!(found, "llama should have a Direct mapping for q_proj");
    }

    #[test]
    fn gpt2_has_fused_c_attn() {
        let decl = family_declaration("gpt2").unwrap();
        let found = decl.mappings.iter().any(|(matcher, mapping)| {
            matches!(matcher, ModuleMatcher::TypeAndName { attr_name, .. } if *attr_name == "c_attn")
                && matches!(mapping, ModuleMapping::Fused { .. })
        });
        assert!(found, "gpt2 should have a Fused mapping for c_attn");
    }

    #[test]
    fn gpt2_c_attn_has_three_equal_splits() {
        let decl = family_declaration("gpt2").unwrap();
        for (matcher, mapping) in decl.mappings {
            if matches!(matcher, ModuleMatcher::TypeAndName { attr_name, .. } if attr_name == "c_attn")
            {
                if let ModuleMapping::Fused { components } = mapping {
                    assert_eq!(components.len(), 3);
                    assert_eq!(components[0].canonical, "q_proj");
                    assert_eq!(components[1].canonical, "k_proj");
                    assert_eq!(components[2].canonical, "v_proj");
                    assert!(components.iter().all(|c| c.split_dim == -1));
                    return;
                }
            }
        }
        panic!("c_attn fused mapping not found");
    }

    #[test]
    fn llama_skip_rotary_emb() {
        let decl = family_declaration("llama").unwrap();
        let found = decl.mappings.iter().any(|(matcher, _)| {
            matches!(matcher, ModuleMatcher::TypeOnly { type_name } if *type_name == "LlamaRotaryEmbedding")
        });
        assert!(found, "llama should have a Skip for LlamaRotaryEmbedding");
    }
}
