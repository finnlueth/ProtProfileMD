use pyo3::prelude::*;

/// Decodes a vector of byte strings into a vector of strings
#[pyfunction]
pub fn decode_bytestrings(byte_strings: Vec<Vec<u8>>) -> PyResult<Vec<String>> {
    byte_strings
        .into_iter()
        .map(|bytes| {
            String::from_utf8(bytes)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyUnicodeDecodeError, _>(
                    format!("Failed to decode bytes to UTF-8: {}", e)
                ))
        })
        .collect()
}