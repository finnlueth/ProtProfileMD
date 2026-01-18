pub mod pdb;
pub mod strings;
pub mod fasta;

// #[pymodule]
// pub fn rust_modules() -> PyResult<()> {
//     m.add_function(wrap_pyfunction!(replace_pdb_coordinates, m)?)?;
//     m.add_function(wrap_pyfunction!(decode_bytestrings, m)?)?;
//     m.add_function(wrap_pyfunction!(strings_to_fasta, m)?)?;
//     Ok(())
// }

use pyo3::prelude::*;
use pdb::replace_pdb_coordinates;
use strings::decode_bytestrings;
use fasta::strings_to_fasta;

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn protprofilemdrust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(replace_pdb_coordinates, m)?)?;
    m.add_function(wrap_pyfunction!(decode_bytestrings, m)?)?;
    m.add_function(wrap_pyfunction!(strings_to_fasta, m)?)?;
    Ok(())
}


