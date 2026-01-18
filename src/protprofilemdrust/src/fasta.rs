use std::fmt::Write;
use pyo3::prelude::*;

/// Converts a vector of strings into FASTA format with numbered frames
#[pyfunction]
pub fn strings_to_fasta(strings: Vec<String>, name: String) -> PyResult<String> {
    let mut fasta = String::new();
    for (i, sequence) in strings.iter().enumerate() {
        writeln!(&mut fasta, ">{}|frame_{}", name, i).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Failed to write header: {}", e)
            )
        })?;
        writeln!(&mut fasta, "{}", sequence.trim()).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Failed to write sequence: {}", e) 
            )
        })?;
    }
    Ok(fasta)
}