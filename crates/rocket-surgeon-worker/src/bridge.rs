#![allow(dead_code)]

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

pub use crate::adapter::{ModelConfig, RawModule};

#[derive(Debug)]
pub struct ModelInfo {
    pub handle: u64,
    pub num_layers: u32,
    pub num_heads: u32,
    pub hidden_dim: u32,
    pub module_tree: Vec<String>,
}

pub fn load_model(source: &str, device: &str, dtype: &str) -> anyhow::Result<u64> {
    Python::with_gil(|py| {
        let skin = py.import("rocket_surgeon.bridge")?;
        let handle = skin
            .getattr("load_model")?
            .call1((source, device, dtype))?
            .extract::<u64>()?;
        Ok(handle)
    })
}

pub fn unload_model(handle: u64) -> anyhow::Result<()> {
    Python::with_gil(|py| {
        let skin = py.import("rocket_surgeon.bridge")?;
        skin.getattr("unload_model")?.call1((handle,))?;
        Ok(())
    })
}

pub fn model_metadata(handle: u64) -> anyhow::Result<ModelInfo> {
    Python::with_gil(|py| {
        let skin = py.import("rocket_surgeon.bridge")?;
        let result = skin.getattr("model_metadata")?.call1((handle,))?;
        let dict = result
            .downcast::<PyDict>()
            .map_err(|e| anyhow::anyhow!("expected dict from model_metadata, got: {e}"))?;

        let num_layers: u32 = dict
            .get_item("num_layers")?
            .ok_or_else(|| anyhow::anyhow!("missing num_layers"))?
            .extract()?;
        let num_heads: u32 = dict
            .get_item("num_heads")?
            .ok_or_else(|| anyhow::anyhow!("missing num_heads"))?
            .extract()?;
        let hidden_dim: u32 = dict
            .get_item("hidden_dim")?
            .ok_or_else(|| anyhow::anyhow!("missing hidden_dim"))?
            .extract()?;
        let module_tree: Vec<String> = dict
            .get_item("module_tree")?
            .ok_or_else(|| anyhow::anyhow!("missing module_tree"))?
            .extract()?;

        Ok(ModelInfo {
            handle,
            num_layers,
            num_heads,
            hidden_dim,
            module_tree,
        })
    })
}

pub fn get_model_config(handle: u64) -> anyhow::Result<ModelConfig> {
    Python::with_gil(|py| {
        let bridge = py.import("rocket_surgeon.bridge")?;
        let result = bridge.getattr("model_config")?.call1((handle,))?;
        let dict = result
            .downcast::<PyDict>()
            .map_err(|e| anyhow::anyhow!("expected dict from model_config, got: {e}"))?;

        let model_type: String = dict
            .get_item("model_type")?
            .ok_or_else(|| anyhow::anyhow!("missing model_type"))?
            .extract()?;
        let num_layers: u32 = dict
            .get_item("num_layers")?
            .ok_or_else(|| anyhow::anyhow!("missing num_layers"))?
            .extract()?;
        let num_heads: u32 = dict
            .get_item("num_heads")?
            .ok_or_else(|| anyhow::anyhow!("missing num_heads"))?
            .extract()?;
        let hidden_size: u32 = dict
            .get_item("hidden_size")?
            .ok_or_else(|| anyhow::anyhow!("missing hidden_size"))?
            .extract()?;
        let num_kv_heads: Option<u32> = dict
            .get_item("num_kv_heads")?
            .and_then(|v| v.extract().ok());

        Ok(ModelConfig {
            model_type,
            num_layers,
            num_heads,
            hidden_size,
            num_kv_heads,
        })
    })
}

pub fn discover_modules(handle: u64) -> anyhow::Result<Vec<RawModule>> {
    Python::with_gil(|py| {
        let bridge = py.import("rocket_surgeon.bridge")?;
        let result = bridge.getattr("discover_modules")?.call1((handle,))?;
        let list = result
            .downcast::<PyList>()
            .map_err(|e| anyhow::anyhow!("expected list from discover_modules, got: {e}"))?;

        let mut modules = Vec::with_capacity(list.len());
        for item in list.iter() {
            let dict = item
                .downcast::<PyDict>()
                .map_err(|e| anyhow::anyhow!("expected dict in modules list, got: {e}"))?;
            let path: String = dict
                .get_item("path")?
                .ok_or_else(|| anyhow::anyhow!("missing path"))?
                .extract()?;
            let type_name: String = dict
                .get_item("type_name")?
                .ok_or_else(|| anyhow::anyhow!("missing type_name"))?
                .extract()?;
            let attr_name: String = dict
                .get_item("attr_name")?
                .ok_or_else(|| anyhow::anyhow!("missing attr_name"))?
                .extract()?;
            modules.push(RawModule {
                path,
                type_name,
                attr_name,
            });
        }
        Ok(modules)
    })
}

pub fn discover_execution_order(handle: u64) -> anyhow::Result<Vec<(String, u32)>> {
    Python::with_gil(|py| {
        let bridge = py.import("rocket_surgeon.bridge")?;
        let result = bridge
            .getattr("discover_execution_order")?
            .call1((handle,))?;
        let list = result
            .downcast::<PyList>()
            .map_err(|e| anyhow::anyhow!("expected list, got: {e}"))?;

        let mut order = Vec::with_capacity(list.len());
        for item in list.iter() {
            let tuple = item
                .downcast::<PyTuple>()
                .map_err(|e| anyhow::anyhow!("expected tuple, got: {e}"))?;
            let path: String = tuple.get_item(0)?.extract()?;
            let call_index: u32 = tuple.get_item(1)?.extract()?;
            order.push((path, call_index));
        }
        Ok(order)
    })
}

pub fn compute_tensor_stats(
    py: Python<'_>,
    tensor: &Bound<'_, pyo3::PyAny>,
) -> anyhow::Result<std::collections::HashMap<String, serde_json::Value>> {
    let bridge = py.import("rocket_surgeon.bridge")?;
    let result = bridge.getattr("compute_tensor_stats")?.call1((tensor,))?;
    let dict = result
        .downcast::<PyDict>()
        .map_err(|e| anyhow::anyhow!("expected dict from compute_tensor_stats, got: {e}"))?;

    let mut stats = std::collections::HashMap::new();
    for (key, value) in dict.iter() {
        let k: String = key.extract()?;
        let v = python_to_json_value(&value)?;
        stats.insert(k, v);
    }
    Ok(stats)
}

fn python_to_json_value(obj: &Bound<'_, pyo3::PyAny>) -> anyhow::Result<serde_json::Value> {
    if let Ok(v) = obj.extract::<i64>() {
        Ok(serde_json::Value::from(v))
    } else if let Ok(v) = obj.extract::<f64>() {
        Ok(serde_json::Value::from(v))
    } else if let Ok(v) = obj.extract::<String>() {
        Ok(serde_json::Value::from(v))
    } else if let Ok(list) = obj.downcast::<PyList>() {
        let items: Vec<serde_json::Value> = list
            .iter()
            .map(|item| python_to_json_value(&item))
            .collect::<anyhow::Result<_>>()?;
        Ok(serde_json::Value::Array(items))
    } else {
        Ok(serde_json::Value::String(obj.str()?.to_string()))
    }
}

pub fn tensor_to_bytes(py: Python<'_>, tensor: &Bound<'_, pyo3::PyAny>) -> anyhow::Result<Vec<u8>> {
    let bridge = py.import("rocket_surgeon.bridge")?;
    let result = bridge.getattr("tensor_to_bytes")?.call1((tensor,))?;
    let bytes: Vec<u8> = result.extract()?;
    Ok(bytes)
}

pub fn split_fused_output<'py>(
    py: Python<'py>,
    tensor: &Bound<'py, pyo3::PyAny>,
    dim: i64,
    sizes: &[usize],
) -> anyhow::Result<Vec<Bound<'py, pyo3::PyAny>>> {
    let bridge = py.import("rocket_surgeon.bridge")?;
    let py_sizes = PyList::new(py, sizes.iter().map(|&s| s as i64))?;
    let result = bridge
        .getattr("split_fused_output")?
        .call1((tensor, dim, py_sizes))?;
    let list = result
        .downcast::<PyList>()
        .map_err(|e| anyhow::anyhow!("expected list, got: {e}"))?;
    Ok(list.iter().collect())
}
