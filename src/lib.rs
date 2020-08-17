#![recursion_limit = "1024"]
#![feature(test)]
#![feature(const_generics, const_generic_impls_guard, fixed_size_array, cow_is_borrowed)]
extern crate test;

#[macro_use]
extern crate anyhow;

pub mod filters;
pub mod idx;
pub mod reader;
