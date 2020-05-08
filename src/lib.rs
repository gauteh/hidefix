#![recursion_limit = "512"] // for Stream reader stream! macro
#![feature(test)]
extern crate test;

#[macro_use]
extern crate anyhow;

pub mod filters;
pub mod idx;
pub mod reader;

