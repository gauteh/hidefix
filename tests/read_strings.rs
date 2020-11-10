#![feature(test)]
extern crate test;

use hidefix::prelude::*;

#[test]
fn chunked_string_array() {
    type T = u8;

    let i = Index::index("tests/data/dmrpp/chunked_string_array.h5").unwrap();
    let mut r = i.reader("string_array").unwrap();

    let values = r.values::<T>(None, None).unwrap();
    let strs = std::str::from_utf8(&values).unwrap();
    println!("{:?}", strs);

    assert_eq!(
        strs,
        "wqwqt\u{0}\u{0}\u{0}jhgjhgjhkjhkjhk\u{0}ddsfdsg\u{0}njiuh\u{0}\u{0}\u{0}"
    );
}
