#![forbid(unsafe_code)]

mod probe_frame;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};

use crate::probe_frame::{HEADER_SIZE, ProbeFrameHeader};

#[pyfunction]
fn blake3_hash(py: Python<'_>, data: &[u8]) -> String {
    py.allow_threads(|| blake3::hash(data).to_hex().to_string())
}

#[pyfunction]
#[pyo3(signature = (rank, layer, comp_id, dtype, ndim, shape, tick_id, data_off, size, flags, generation))]
#[allow(clippy::too_many_arguments, clippy::needless_pass_by_value)]
fn serialize_probe_frame_header(
    py: Python<'_>,
    rank: u32,
    layer: u32,
    comp_id: u16,
    dtype: u8,
    ndim: u8,
    shape: Vec<u32>,
    tick_id: u64,
    data_off: u64,
    size: u64,
    flags: u32,
    generation: u32,
) -> PyResult<Py<PyBytes>> {
    if shape.len() > 8 {
        return Err(PyValueError::new_err(format!(
            "shape has {} dims, max is 8",
            shape.len()
        )));
    }
    if usize::from(ndim) != shape.len() {
        return Err(PyValueError::new_err(format!(
            "ndim ({ndim}) does not match shape length ({})",
            shape.len()
        )));
    }

    let mut shape_arr = [0u32; 8];
    for (i, &dim) in shape.iter().enumerate() {
        shape_arr[i] = dim;
    }

    let header = ProbeFrameHeader {
        rank,
        layer,
        comp_id,
        dtype,
        ndim,
        shape: shape_arr,
        tick_id,
        data_off,
        size,
        flags,
        generation,
    };

    let bytes = header.serialize();
    Ok(PyBytes::new(py, &bytes).into())
}

#[pyfunction]
fn parse_probe_frame_header<'py>(py: Python<'py>, data: &[u8]) -> PyResult<Bound<'py, PyDict>> {
    let header = ProbeFrameHeader::parse(data).map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item("rank", header.rank)?;
    dict.set_item("layer", header.layer)?;
    dict.set_item("comp_id", header.comp_id)?;
    dict.set_item("dtype", header.dtype)?;
    dict.set_item("ndim", header.ndim)?;
    dict.set_item("shape", PyList::new(py, header.shape)?)?;
    dict.set_item("tick_id", header.tick_id)?;
    dict.set_item("data_off", header.data_off)?;
    dict.set_item("size", header.size)?;
    dict.set_item("flags", header.flags)?;
    dict.set_item("generation", header.generation)?;
    Ok(dict)
}

#[pymodule]
fn _rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("PROBE_FRAME_HEADER_SIZE", HEADER_SIZE)?;
    m.add_function(wrap_pyfunction!(blake3_hash, m)?)?;
    m.add_function(wrap_pyfunction!(serialize_probe_frame_header, m)?)?;
    m.add_function(wrap_pyfunction!(parse_probe_frame_header, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::probe_frame::ProbeFrameHeader;

    const BLAKE3_EMPTY: &str = "af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262";

    #[test]
    fn blake3_hash_empty() {
        let hash = blake3::hash(b"").to_hex().to_string();
        assert_eq!(hash, BLAKE3_EMPTY);
    }

    #[test]
    fn blake3_hash_known_input() {
        let hash = blake3::hash(b"hello").to_hex().to_string();
        assert_eq!(hash.len(), 64);
        assert_ne!(hash, BLAKE3_EMPTY);
        let hash2 = blake3::hash(b"hello").to_hex().to_string();
        assert_eq!(hash, hash2);
    }

    #[test]
    fn blake3_deterministic() {
        let data = vec![0u8; 1024];
        let h1 = blake3::hash(&data).to_hex().to_string();
        let h2 = blake3::hash(&data).to_hex().to_string();
        assert_eq!(h1, h2);
    }

    #[test]
    fn probe_frame_round_trip_via_module() {
        let header = ProbeFrameHeader {
            rank: 2,
            layer: 15,
            comp_id: 7,
            dtype: 2,
            ndim: 2,
            shape: [4096, 4096, 0, 0, 0, 0, 0, 0],
            tick_id: 100,
            data_off: 0,
            size: 4096 * 4096 * 4,
            flags: 0,
            generation: 0,
        };
        let bytes = header.serialize();
        assert_eq!(bytes.len(), 128);
        let parsed = ProbeFrameHeader::parse(&bytes).unwrap();
        assert_eq!(header, parsed);
    }
}
