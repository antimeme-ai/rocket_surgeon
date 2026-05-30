#![allow(dead_code)]

use pyo3::IntoPyObjectExt;
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

pub fn collect_export_env(handle: u64) -> anyhow::Result<serde_json::Value> {
    Python::with_gil(|py| {
        let bridge = py.import("rocket_surgeon.bridge")?;
        let result = bridge.getattr("collect_export_env")?.call1((handle,))?;
        let json_str = py
            .import("json")?
            .getattr("dumps")?
            .call1((result,))?
            .extract::<String>()?;
        let value: serde_json::Value = serde_json::from_str(&json_str)?;
        Ok(value)
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

pub fn get_tensor_shape(
    _py: Python<'_>,
    tensor: &Bound<'_, pyo3::PyAny>,
) -> anyhow::Result<Vec<u64>> {
    let shape = tensor.getattr("shape")?;
    let dims: Vec<u64> = shape
        .try_iter()?
        .map(|d| d.unwrap().extract().unwrap())
        .collect();
    Ok(dims)
}

pub fn get_tensor_dtype(
    _py: Python<'_>,
    tensor: &Bound<'_, pyo3::PyAny>,
) -> anyhow::Result<String> {
    let dtype = tensor.getattr("dtype")?;
    let name = dtype.str()?.to_string();
    let clean = name.strip_prefix("torch.").unwrap_or(&name);
    Ok(clean.to_owned())
}

pub fn get_tensor_device(
    _py: Python<'_>,
    tensor: &Bound<'_, pyo3::PyAny>,
) -> anyhow::Result<String> {
    let device = tensor.getattr("device")?;
    let s = device.str()?.to_string();
    Ok(s)
}

pub fn install_sentinel_hooks(
    handle: u64,
    module_paths: &[String],
) -> anyhow::Result<Vec<pyo3::PyObject>> {
    Python::with_gil(|py| {
        let bridge = py.import("rocket_surgeon.bridge")?;
        let py_paths = PyList::new(py, module_paths)?;
        let result = bridge
            .getattr("install_sentinel_hooks")?
            .call1((handle, py_paths))?;
        let list = result
            .downcast::<PyList>()
            .map_err(|e| anyhow::anyhow!("expected list, got: {e}"))?;
        let handles: Vec<pyo3::PyObject> =
            list.iter().map(|h| h.into_py_any(py).unwrap()).collect();
        Ok(handles)
    })
}

pub fn install_passive_hooks<'py>(
    py: Python<'py>,
    handle: u64,
    container_paths: &[String],
    last_outputs: &Bound<'py, pyo3::PyAny>,
) -> anyhow::Result<Vec<pyo3::PyObject>> {
    let bridge = py.import("rocket_surgeon.bridge")?;
    let py_paths = PyList::new(py, container_paths)?;
    let result =
        bridge
            .getattr("install_passive_hooks")?
            .call1((handle, py_paths, last_outputs))?;
    let list = result
        .downcast::<PyList>()
        .map_err(|e| anyhow::anyhow!("expected list from install_passive_hooks, got: {e}"))?;
    let handles: Vec<pyo3::PyObject> = list.iter().map(|h| h.into_py_any(py).unwrap()).collect();
    Ok(handles)
}

pub fn install_capture_hooks<'py>(
    py: Python<'py>,
    handle: u64,
    module_paths: &[String],
    result_mailbox: &Bound<'py, pyo3::PyAny>,
    resume_mailbox: &Bound<'py, pyo3::PyAny>,
    active_probes: &[String],
) -> anyhow::Result<(Vec<pyo3::PyObject>, Bound<'py, PyDict>)> {
    let bridge = py.import("rocket_surgeon.bridge")?;
    let py_paths = PyList::new(py, module_paths)?;
    let py_probes: pyo3::Bound<'_, pyo3::types::PySet> =
        pyo3::types::PySet::new(py, active_probes)?;
    let result = bridge.getattr("install_capture_hooks")?.call1((
        handle,
        py_paths,
        result_mailbox,
        resume_mailbox,
        py_probes,
    ))?;
    let tuple = result
        .downcast::<PyTuple>()
        .map_err(|e| anyhow::anyhow!("expected tuple, got: {e}"))?;
    let item0 = tuple.get_item(0)?;
    let hook_list = item0
        .downcast::<PyList>()
        .map_err(|e| anyhow::anyhow!("expected list for handles, got: {e}"))?;
    let item1 = tuple.get_item(1)?;
    let call_counts = item1
        .downcast::<PyDict>()
        .map_err(|e| anyhow::anyhow!("expected dict for call_counts, got: {e}"))?;
    let handles: Vec<pyo3::PyObject> = hook_list
        .iter()
        .map(|h| h.into_py_any(py).unwrap())
        .collect();
    Ok((handles, call_counts.clone()))
}

pub fn run_forward(
    py: Python<'_>,
    handle: u64,
    input_ids: Option<&[u64]>,
    done_callback: &Bound<'_, pyo3::PyAny>,
) -> anyhow::Result<()> {
    let bridge = py.import("rocket_surgeon.bridge")?;
    let torch = py.import("torch")?;

    let py_input = if let Some(ids) = input_ids {
        let py_list = PyList::new(py, ids.iter().map(|&id| id as i64))?;
        let tensor = torch.getattr("tensor")?.call1((vec![py_list],))?;
        tensor.call_method1("to", (torch.getattr("long")?,))?
    } else {
        let zeros = torch.getattr("zeros")?.call1(((1, 2),))?;
        zeros.call_method1("to", (torch.getattr("long")?,))?
    };

    bridge
        .getattr("run_forward")?
        .call1((handle, py_input, done_callback))?;
    Ok(())
}

pub fn remove_hooks(py: Python<'_>, handles: &[pyo3::PyObject]) -> anyhow::Result<()> {
    let bridge = py.import("rocket_surgeon.bridge")?;
    let py_handles = PyList::new(py, handles.iter().map(|h| h.bind(py)))?;
    bridge.getattr("remove_hooks")?.call1((py_handles,))?;
    Ok(())
}

pub fn create_mailbox(py: Python<'_>) -> anyhow::Result<pyo3::PyObject> {
    let mailbox_mod = py.import("rocket_surgeon.hooks.mailbox")?;
    let mb = mailbox_mod.getattr("Mailbox")?.call0()?;
    Ok(mb.into_py_any(py)?)
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

pub fn register_arena_cuda(ptr: usize, size: usize) -> anyhow::Result<bool> {
    Python::with_gil(|py| {
        let ckpt = py.import("rocket_surgeon.checkpoint")?;
        let result: bool = ckpt
            .getattr("register_cuda_pinned")?
            .call1((ptr, size))?
            .extract()?;
        Ok(result)
    })
}

pub fn unregister_arena_cuda(ptr: usize) -> anyhow::Result<bool> {
    Python::with_gil(|py| {
        let ckpt = py.import("rocket_surgeon.checkpoint")?;
        let result: bool = ckpt
            .getattr("unregister_cuda_pinned")?
            .call1((ptr,))?
            .extract()?;
        Ok(result)
    })
}

pub fn activation_available(
    py: Python<'_>,
    last_outputs: &pyo3::PyObject,
    component_path: &str,
    call_index: u32,
) -> anyhow::Result<bool> {
    let ckpt = py.import("rocket_surgeon.checkpoint")?;
    let result: bool = ckpt
        .getattr("activation_available")?
        .call1((last_outputs.bind(py), component_path, call_index))?
        .extract()?;
    Ok(result)
}

pub fn capture_activation(
    py: Python<'_>,
    last_outputs: &pyo3::PyObject,
    component_path: &str,
    call_index: u32,
    dst_ptr: usize,
    dst_len: usize,
) -> anyhow::Result<(String, Vec<i64>)> {
    let ckpt = py.import("rocket_surgeon.checkpoint")?;
    let result = ckpt.getattr("capture_activation")?.call1((
        last_outputs.bind(py),
        component_path,
        call_index,
        dst_ptr,
        dst_len,
    ))?;
    let tuple = result
        .downcast::<PyTuple>()
        .map_err(|e| anyhow::anyhow!("expected tuple: {e}"))?;
    let dtype: String = tuple.get_item(0)?.extract()?;
    let shape: Vec<i64> = tuple.get_item(1)?.extract()?;
    Ok((dtype, shape))
}

#[allow(clippy::too_many_arguments)]
pub fn restore_activation(
    py: Python<'_>,
    last_outputs: &pyo3::PyObject,
    component_path: &str,
    call_index: u32,
    src_ptr: usize,
    src_len: usize,
    dtype: &str,
    shape: &[i64],
) -> anyhow::Result<()> {
    let ckpt = py.import("rocket_surgeon.checkpoint")?;
    let shape_list = PyList::new(py, shape)?;
    ckpt.getattr("restore_activation")?.call1((
        last_outputs.bind(py),
        component_path,
        call_index,
        src_ptr,
        src_len,
        dtype,
        shape_list,
    ))?;
    Ok(())
}

pub fn capture_rng_state() -> anyhow::Result<Vec<u8>> {
    Python::with_gil(|py| {
        let ckpt = py.import("rocket_surgeon.checkpoint")?;
        let result: Vec<u8> = ckpt.getattr("capture_rng_state")?.call0()?.extract()?;
        Ok(result)
    })
}

pub fn restore_rng_state(state: &[u8]) -> anyhow::Result<()> {
    Python::with_gil(|py| {
        let ckpt = py.import("rocket_surgeon.checkpoint")?;
        ckpt.getattr("restore_rng_state")?.call1((state,))?;
        Ok(())
    })
}

pub fn capture_cpu_rng_state() -> anyhow::Result<Vec<u8>> {
    Python::with_gil(|py| {
        let ckpt = py.import("rocket_surgeon.checkpoint")?;
        let result: Vec<u8> = ckpt.getattr("capture_cpu_rng_state")?.call0()?.extract()?;
        Ok(result)
    })
}

pub fn restore_cpu_rng_state(state: &[u8]) -> anyhow::Result<()> {
    Python::with_gil(|py| {
        let ckpt = py.import("rocket_surgeon.checkpoint")?;
        let py_bytes = pyo3::types::PyBytes::new(py, state);
        ckpt.getattr("restore_cpu_rng_state")?.call1((py_bytes,))?;
        Ok(())
    })
}

#[allow(clippy::too_many_arguments)]
pub fn compare_activations_from_ptr(
    py: Python<'_>,
    original_ptr: usize,
    original_len: usize,
    original_dtype: &str,
    original_shape: &[i64],
    replayed: &Bound<'_, pyo3::PyAny>,
    cosine_threshold: f64,
    mre_threshold: f64,
) -> anyhow::Result<Option<(f64, f64)>> {
    let replay_mod = py.import("rocket_surgeon.replay")?;
    let py_shape = PyList::new(py, original_shape)?;
    let result = replay_mod.getattr("compare_activations_from_ptr")?.call1((
        original_ptr,
        original_len,
        original_dtype,
        py_shape,
        replayed,
        cosine_threshold,
        mre_threshold,
    ))?;

    if result.is_none() {
        return Ok(None);
    }
    let dict = result
        .downcast::<PyDict>()
        .map_err(|e| anyhow::anyhow!("expected dict from compare_activations_from_ptr: {e}"))?;
    let cosine: f64 = dict
        .get_item("cosine_similarity")?
        .ok_or_else(|| anyhow::anyhow!("missing cosine_similarity key"))?
        .extract()?;
    let mre: f64 = dict
        .get_item("max_relative_error")?
        .ok_or_else(|| anyhow::anyhow!("missing max_relative_error key"))?
        .extract()?;
    Ok(Some((cosine, mre)))
}

#[allow(clippy::too_many_arguments)]
pub fn apply_interventions_at_point<'py>(
    py: Python<'py>,
    tensor: &Bound<'py, pyo3::PyAny>,
    recipes_json: &str,
    family: &str,
    rank: u32,
    layer: u32,
    component: &str,
    event: &str,
    tick_id: u64,
    model_handle: u64,
) -> anyhow::Result<(Bound<'py, pyo3::PyAny>, Vec<String>)> {
    let bridge = py.import("rocket_surgeon.bridge")?;
    let json_mod = py.import("json")?;
    let py_recipes = json_mod.getattr("loads")?.call1((recipes_json,))?;
    let result = bridge.getattr("apply_interventions_at_point")?.call1((
        tensor,
        py_recipes,
        family,
        rank,
        layer,
        component,
        event,
        tick_id,
        model_handle,
    ))?;
    let tuple = result.downcast::<PyTuple>().map_err(|e| {
        anyhow::anyhow!("expected tuple from apply_interventions_at_point, got: {e}")
    })?;
    let modified_tensor = tuple.get_item(0)?;
    let fired_list = tuple.get_item(1)?;
    let fired: Vec<String> = fired_list
        .downcast::<PyList>()
        .map_err(|e| anyhow::anyhow!("expected list for fired_ids, got: {e}"))?
        .iter()
        .map(|item| item.extract::<String>())
        .collect::<PyResult<_>>()?;
    Ok((modified_tensor, fired))
}
