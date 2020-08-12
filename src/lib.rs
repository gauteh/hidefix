#![recursion_limit = "1024"]
#![feature(test)]
#![feature(const_generics)]
extern crate test;

#[macro_use]
extern crate anyhow;

pub mod filters;
pub mod idx;
pub mod reader;
