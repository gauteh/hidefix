#[macro_use]
extern crate anyhow;

pub mod idx;
pub mod reader;

#[cfg(feature = "io_uring")]
pub mod uring;

#[cfg(test)]
mod tests {}
