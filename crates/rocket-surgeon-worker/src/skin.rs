use pyo3::prelude::*;
use pyo3::types::PyDict;

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
        let skin = py.import("rocket_surgeon.skin")?;
        let handle = skin
            .getattr("load_model")?
            .call1((source, device, dtype))?
            .extract::<u64>()?;
        Ok(handle)
    })
}

pub fn unload_model(handle: u64) -> anyhow::Result<()> {
    Python::with_gil(|py| {
        let skin = py.import("rocket_surgeon.skin")?;
        skin.getattr("unload_model")?.call1((handle,))?;
        Ok(())
    })
}

pub fn model_metadata(handle: u64) -> anyhow::Result<ModelInfo> {
    Python::with_gil(|py| {
        let skin = py.import("rocket_surgeon.skin")?;
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
