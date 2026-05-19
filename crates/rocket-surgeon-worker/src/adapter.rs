#![allow(dead_code)]

use std::collections::{HashMap, HashSet};

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
        path_contains: Option<&'static str>,
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
                    path_contains: None,
                },
                Direct {
                    canonical: "ln1".to_owned(),
                },
            ),
            (
                TypeAndName {
                    type_name: "LlamaRMSNorm",
                    attr_name: "post_attention_layernorm",
                    path_contains: None,
                },
                Direct {
                    canonical: "ln2".to_owned(),
                },
            ),
            (
                TypeAndName {
                    type_name: "LlamaRMSNorm",
                    attr_name: "norm",
                    path_contains: None,
                },
                Direct {
                    canonical: "ln_final".to_owned(),
                },
            ),
            (
                TypeAndName {
                    type_name: "MistralRMSNorm",
                    attr_name: "input_layernorm",
                    path_contains: None,
                },
                Direct {
                    canonical: "ln1".to_owned(),
                },
            ),
            (
                TypeAndName {
                    type_name: "MistralRMSNorm",
                    attr_name: "post_attention_layernorm",
                    path_contains: None,
                },
                Direct {
                    canonical: "ln2".to_owned(),
                },
            ),
            (
                TypeAndName {
                    type_name: "MistralRMSNorm",
                    attr_name: "norm",
                    path_contains: None,
                },
                Direct {
                    canonical: "ln_final".to_owned(),
                },
            ),
            (
                TypeAndName {
                    type_name: "Linear",
                    attr_name: "q_proj",
                    path_contains: None,
                },
                Direct {
                    canonical: "q_proj".to_owned(),
                },
            ),
            (
                TypeAndName {
                    type_name: "Linear",
                    attr_name: "k_proj",
                    path_contains: None,
                },
                Direct {
                    canonical: "k_proj".to_owned(),
                },
            ),
            (
                TypeAndName {
                    type_name: "Linear",
                    attr_name: "v_proj",
                    path_contains: None,
                },
                Direct {
                    canonical: "v_proj".to_owned(),
                },
            ),
            (
                TypeAndName {
                    type_name: "Linear",
                    attr_name: "o_proj",
                    path_contains: None,
                },
                Direct {
                    canonical: "o_proj".to_owned(),
                },
            ),
            (
                TypeAndName {
                    type_name: "Linear",
                    attr_name: "gate_proj",
                    path_contains: None,
                },
                Direct {
                    canonical: "gate_proj".to_owned(),
                },
            ),
            (
                TypeAndName {
                    type_name: "Linear",
                    attr_name: "up_proj",
                    path_contains: None,
                },
                Direct {
                    canonical: "up_proj".to_owned(),
                },
            ),
            (
                TypeAndName {
                    type_name: "Linear",
                    attr_name: "down_proj",
                    path_contains: None,
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
                    path_contains: None,
                },
                Direct {
                    canonical: "embed".to_owned(),
                },
            ),
            (
                TypeAndName {
                    type_name: "Linear",
                    attr_name: "lm_head",
                    path_contains: None,
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
                    path_contains: None,
                },
                Direct {
                    canonical: "ln1".to_owned(),
                },
            ),
            (
                TypeAndName {
                    type_name: "LayerNorm",
                    attr_name: "ln_2",
                    path_contains: None,
                },
                Direct {
                    canonical: "ln2".to_owned(),
                },
            ),
            (
                TypeAndName {
                    type_name: "LayerNorm",
                    attr_name: "ln_f",
                    path_contains: None,
                },
                Direct {
                    canonical: "ln_final".to_owned(),
                },
            ),
            (
                TypeAndName {
                    type_name: "Conv1D",
                    attr_name: "c_attn",
                    path_contains: None,
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
                    path_contains: Some("attn"),
                },
                Direct {
                    canonical: "o_proj".to_owned(),
                },
            ),
            (
                TypeAndName {
                    type_name: "Conv1D",
                    attr_name: "c_fc",
                    path_contains: None,
                },
                Direct {
                    canonical: "up_proj".to_owned(),
                },
            ),
            (
                TypeAndName {
                    type_name: "Conv1D",
                    attr_name: "c_proj",
                    path_contains: Some("mlp"),
                },
                Direct {
                    canonical: "down_proj".to_owned(),
                },
            ),
            (
                TypeAndName {
                    type_name: "Embedding",
                    attr_name: "wte",
                    path_contains: None,
                },
                Direct {
                    canonical: "embed".to_owned(),
                },
            ),
            (
                TypeAndName {
                    type_name: "Embedding",
                    attr_name: "wpe",
                    path_contains: None,
                },
                Direct {
                    canonical: "pos_embed".to_owned(),
                },
            ),
            (
                TypeAndName {
                    type_name: "Linear",
                    attr_name: "lm_head",
                    path_contains: None,
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

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RawModule {
    pub path: String,
    pub type_name: String,
    pub attr_name: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelConfig {
    pub model_type: String,
    pub num_layers: u32,
    pub num_heads: u32,
    pub hidden_size: u32,
    pub num_kv_heads: Option<u32>,
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

fn extract_layer_index(path: &str) -> Option<u32> {
    for segment in path.split('.') {
        if let Ok(idx) = segment.parse::<u32>() {
            return Some(idx);
        }
    }
    None
}

fn matches_module(matcher: &ModuleMatcher, module: &RawModule) -> bool {
    match matcher {
        ModuleMatcher::TypeOnly { type_name } => module.type_name == *type_name,
        ModuleMatcher::TypeAndName {
            type_name,
            attr_name,
            path_contains,
        } => {
            module.type_name == *type_name
                && module.attr_name == *attr_name
                && path_contains.is_none_or(|pat| module.path.contains(pat))
        }
    }
}

pub fn resolve(
    modules: &[RawModule],
    config: &ModelConfig,
    rank: u32,
) -> Result<ComponentMap, String> {
    resolve_with_containers(modules, config, rank).map(|(map, _)| map)
}

pub fn resolve_with_containers(
    modules: &[RawModule],
    config: &ModelConfig,
    rank: u32,
) -> Result<(ComponentMap, Vec<String>), String> {
    let decl = family_declaration(&config.model_type)
        .ok_or_else(|| format!("unsupported model family: {}", config.model_type))?;

    let family_name = config.model_type.clone();
    let mut components = Vec::new();
    let mut vocabulary = HashSet::new();
    let mut container_paths = Vec::new();

    for module in modules {
        let mut matched = false;

        for (matcher, mapping) in &decl.mappings {
            if matches_module(matcher, module) {
                matched = true;
                match mapping {
                    ModuleMapping::Skip => break,
                    ModuleMapping::Container => {
                        container_paths.push(module.path.clone());
                        break;
                    }
                    ModuleMapping::Direct { canonical } => {
                        let layer_index = extract_layer_index(&module.path);
                        vocabulary.insert(canonical.clone());
                        let layer = layer_index.unwrap_or(0);
                        components.push(MappedComponent {
                            module_path: module.path.clone(),
                            canonical: canonical.clone(),
                            layer_index,
                            call_index: 0,
                            mapping: mapping.clone(),
                            probe_point: format!("model:{rank}:{layer}:{canonical}:0:fwd"),
                        });
                        break;
                    }
                    ModuleMapping::Fused {
                        components: fused_comps,
                    } => {
                        let layer_index = extract_layer_index(&module.path);
                        let mut resolved_fused = fused_comps.clone();
                        for fc in &mut resolved_fused {
                            if fc.split_size == 0 {
                                fc.split_size = config.hidden_size as usize;
                            }
                            vocabulary.insert(fc.canonical.clone());
                        }
                        let layer = layer_index.unwrap_or(0);
                        let fused_canonical = format!("_fused.{}", module.attr_name);
                        components.push(MappedComponent {
                            module_path: module.path.clone(),
                            canonical: fused_canonical.clone(),
                            layer_index,
                            call_index: 0,
                            mapping: ModuleMapping::Fused {
                                components: resolved_fused,
                            },
                            probe_point: format!("model:{rank}:{layer}:{fused_canonical}:0:fwd"),
                        });
                        break;
                    }
                }
            }
        }

        if !matched {
            let canonical = format!("_raw.{}", module.path);
            let layer_index = extract_layer_index(&module.path);
            let layer = layer_index.unwrap_or(0);
            vocabulary.insert(canonical.clone());
            components.push(MappedComponent {
                module_path: module.path.clone(),
                canonical: canonical.clone(),
                layer_index,
                call_index: 0,
                mapping: ModuleMapping::Direct {
                    canonical: format!("_raw.{}", module.path),
                },
                probe_point: format!("model:{rank}:{layer}:{canonical}:0:fwd"),
            });
        }
    }

    let mut vocab_sorted: Vec<String> = vocabulary.into_iter().collect();
    vocab_sorted.sort();

    Ok((
        ComponentMap {
            components,
            model_family: family_name,
            vocabulary: vocab_sorted,
        },
        container_paths,
    ))
}

pub fn apply_execution_order(map: &mut ComponentMap, execution_order: &[(String, u32)]) {
    let order_map: HashMap<(&str, u32), usize> = execution_order
        .iter()
        .enumerate()
        .map(|(i, (path, ci))| ((path.as_str(), *ci), i))
        .collect();

    map.components.sort_by_key(|c| {
        order_map
            .get(&(c.module_path.as_str(), c.call_index))
            .copied()
            .unwrap_or(usize::MAX)
    });
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

    #[test]
    fn resolve_llama_modules() {
        let modules = vec![
            RawModule {
                path: "model".into(),
                type_name: "LlamaModel".into(),
                attr_name: "model".into(),
            },
            RawModule {
                path: "model.embed_tokens".into(),
                type_name: "Embedding".into(),
                attr_name: "embed_tokens".into(),
            },
            RawModule {
                path: "model.layers.0.self_attn".into(),
                type_name: "LlamaSdpaAttention".into(),
                attr_name: "self_attn".into(),
            },
            RawModule {
                path: "model.layers.0.self_attn.q_proj".into(),
                type_name: "Linear".into(),
                attr_name: "q_proj".into(),
            },
            RawModule {
                path: "model.layers.0.self_attn.k_proj".into(),
                type_name: "Linear".into(),
                attr_name: "k_proj".into(),
            },
            RawModule {
                path: "model.layers.0.self_attn.v_proj".into(),
                type_name: "Linear".into(),
                attr_name: "v_proj".into(),
            },
            RawModule {
                path: "model.layers.0.self_attn.o_proj".into(),
                type_name: "Linear".into(),
                attr_name: "o_proj".into(),
            },
            RawModule {
                path: "model.layers.0.input_layernorm".into(),
                type_name: "LlamaRMSNorm".into(),
                attr_name: "input_layernorm".into(),
            },
            RawModule {
                path: "model.layers.0.mlp".into(),
                type_name: "LlamaMLP".into(),
                attr_name: "mlp".into(),
            },
            RawModule {
                path: "model.layers.0.mlp.gate_proj".into(),
                type_name: "Linear".into(),
                attr_name: "gate_proj".into(),
            },
            RawModule {
                path: "model.layers.0.mlp.up_proj".into(),
                type_name: "Linear".into(),
                attr_name: "up_proj".into(),
            },
            RawModule {
                path: "model.layers.0.mlp.down_proj".into(),
                type_name: "Linear".into(),
                attr_name: "down_proj".into(),
            },
            RawModule {
                path: "model.layers.0.post_attention_layernorm".into(),
                type_name: "LlamaRMSNorm".into(),
                attr_name: "post_attention_layernorm".into(),
            },
            RawModule {
                path: "lm_head".into(),
                type_name: "Linear".into(),
                attr_name: "lm_head".into(),
            },
        ];
        let config = ModelConfig {
            model_type: "llama".into(),
            num_layers: 1,
            num_heads: 4,
            hidden_size: 32,
            num_kv_heads: Some(4),
        };
        let map = resolve(&modules, &config, 0).unwrap();
        assert_eq!(map.model_family, "llama");

        let canonicals: Vec<&str> = map
            .components
            .iter()
            .map(|c| c.canonical.as_str())
            .collect();
        assert!(canonicals.contains(&"q_proj"));
        assert!(canonicals.contains(&"k_proj"));
        assert!(canonicals.contains(&"v_proj"));
        assert!(canonicals.contains(&"o_proj"));
        assert!(canonicals.contains(&"gate_proj"));
        assert!(canonicals.contains(&"up_proj"));
        assert!(canonicals.contains(&"down_proj"));
        assert!(canonicals.contains(&"ln1"));
        assert!(canonicals.contains(&"ln2"));
        assert!(canonicals.contains(&"embed"));
        assert!(canonicals.contains(&"lm_head"));
    }

    #[test]
    fn resolve_detects_layer_index() {
        let modules = vec![RawModule {
            path: "model.layers.3.self_attn.q_proj".into(),
            type_name: "Linear".into(),
            attr_name: "q_proj".into(),
        }];
        let config = ModelConfig {
            model_type: "llama".into(),
            num_layers: 4,
            num_heads: 4,
            hidden_size: 32,
            num_kv_heads: Some(4),
        };
        let map = resolve(&modules, &config, 0).unwrap();
        let q = map
            .components
            .iter()
            .find(|c| c.canonical == "q_proj")
            .unwrap();
        assert_eq!(q.layer_index, Some(3));
    }

    #[test]
    fn resolve_unknown_module_gets_raw_fallback() {
        let modules = vec![RawModule {
            path: "model.weird_thing".into(),
            type_name: "UnknownModule".into(),
            attr_name: "weird_thing".into(),
        }];
        let config = ModelConfig {
            model_type: "llama".into(),
            num_layers: 1,
            num_heads: 4,
            hidden_size: 32,
            num_kv_heads: Some(4),
        };
        let map = resolve(&modules, &config, 0).unwrap();
        let raw = map
            .components
            .iter()
            .find(|c| c.canonical.starts_with("_raw."))
            .unwrap();
        assert_eq!(raw.canonical, "_raw.model.weird_thing");
    }

    #[test]
    fn resolve_unsupported_family_returns_error() {
        let modules = vec![];
        let config = ModelConfig {
            model_type: "unknown_arch".into(),
            num_layers: 1,
            num_heads: 4,
            hidden_size: 32,
            num_kv_heads: None,
        };
        let result = resolve(&modules, &config, 0);
        assert!(result.is_err());
    }

    #[test]
    fn resolve_builds_vocabulary() {
        let modules = vec![
            RawModule {
                path: "model.layers.0.self_attn.q_proj".into(),
                type_name: "Linear".into(),
                attr_name: "q_proj".into(),
            },
            RawModule {
                path: "model.layers.0.self_attn.k_proj".into(),
                type_name: "Linear".into(),
                attr_name: "k_proj".into(),
            },
        ];
        let config = ModelConfig {
            model_type: "llama".into(),
            num_layers: 1,
            num_heads: 4,
            hidden_size: 32,
            num_kv_heads: Some(4),
        };
        let map = resolve(&modules, &config, 0).unwrap();
        assert!(map.vocabulary.contains(&"q_proj".to_owned()));
        assert!(map.vocabulary.contains(&"k_proj".to_owned()));
    }

    #[test]
    fn apply_execution_order_reorders_components() {
        let mut map = ComponentMap {
            components: vec![
                MappedComponent {
                    module_path: "a".into(),
                    canonical: "first".into(),
                    layer_index: Some(0),
                    call_index: 0,
                    mapping: ModuleMapping::Direct {
                        canonical: "first".into(),
                    },
                    probe_point: String::new(),
                },
                MappedComponent {
                    module_path: "b".into(),
                    canonical: "second".into(),
                    layer_index: Some(0),
                    call_index: 0,
                    mapping: ModuleMapping::Direct {
                        canonical: "second".into(),
                    },
                    probe_point: String::new(),
                },
            ],
            model_family: "test".into(),
            vocabulary: vec![],
        };
        let execution_order = vec![("b".to_owned(), 0u32), ("a".to_owned(), 0u32)];
        apply_execution_order(&mut map, &execution_order);
        assert_eq!(map.components[0].module_path, "b");
        assert_eq!(map.components[1].module_path, "a");
    }

    #[test]
    fn container_paths_includes_llama_decoder_layers() {
        let modules = vec![
            RawModule {
                path: "model.layers.0".into(),
                type_name: "LlamaDecoderLayer".into(),
                attr_name: "0".into(),
            },
            RawModule {
                path: "model.layers.0.self_attn".into(),
                type_name: "LlamaSdpaAttention".into(),
                attr_name: "self_attn".into(),
            },
            RawModule {
                path: "model.layers.0.mlp".into(),
                type_name: "LlamaMLP".into(),
                attr_name: "mlp".into(),
            },
            RawModule {
                path: "model.layers.0.self_attn.q_proj".into(),
                type_name: "Linear".into(),
                attr_name: "q_proj".into(),
            },
        ];
        let config = ModelConfig {
            model_type: "llama".into(),
            num_layers: 1,
            num_heads: 4,
            hidden_size: 64,
            num_kv_heads: Some(4),
        };
        let (_, containers) = resolve_with_containers(&modules, &config, 0).unwrap();
        assert!(containers.contains(&"model.layers.0".to_owned()));
        assert!(containers.contains(&"model.layers.0.self_attn".to_owned()));
        assert!(containers.contains(&"model.layers.0.mlp".to_owned()));
        assert!(!containers.contains(&"model.layers.0.self_attn.q_proj".to_owned()));
    }

    #[test]
    fn container_paths_empty_for_skip_modules() {
        let modules = vec![
            RawModule {
                path: "model.embed_tokens".into(),
                type_name: "Embedding".into(),
                attr_name: "embed_tokens".into(),
            },
            RawModule {
                path: "model.rotary_emb".into(),
                type_name: "LlamaRotaryEmbedding".into(),
                attr_name: "rotary_emb".into(),
            },
        ];
        let config = ModelConfig {
            model_type: "llama".into(),
            num_layers: 1,
            num_heads: 4,
            hidden_size: 64,
            num_kv_heads: Some(4),
        };
        let (_, containers) = resolve_with_containers(&modules, &config, 0).unwrap();
        assert!(containers.is_empty());
    }
}
