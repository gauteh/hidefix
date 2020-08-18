#![allow(incomplete_features)]
#![recursion_limit = "1024"]
#![feature(test)]
#![feature(const_generics, fixed_size_array, cow_is_borrowed)]
extern crate test;

#[macro_use]
extern crate anyhow;

pub mod filters;
pub mod idx;
pub mod reader;
