#![feature(vec_into_raw_parts)]
#![feature(test)]
extern crate test;

#[macro_use]
extern crate anyhow;

pub mod filters;
pub mod idx;
pub mod reader;

#[cfg(test)]
mod tests {}
