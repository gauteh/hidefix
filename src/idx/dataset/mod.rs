mod any;
#[allow(clippy::module_inception)]
mod dataset;
mod types;

pub use any::*;
pub use dataset::*;
pub use types::*;

#[cfg(test)]
mod tests {
    use super::super::chunk::{Chunk, ULE};
    use super::*;
    use crate::filters::byteorder::Order as ByteOrder;
    use itertools::izip;
    use serde::{Deserialize, Serialize};
    use test::Bencher;

    pub(crate) fn test_dataset() -> Dataset<'static, 2> {
        Dataset::new(
            Datatype::Float(2),
            ByteOrder::BE,
            [20, 20],
            vec![
                Chunk::new(0, 400, [0, 0]),
                Chunk::new(400, 400, [0, 10]),
                Chunk::new(800, 400, [10, 0]),
                Chunk::new(1200, 400, [10, 10]),
            ],
            [10, 10],
            false,
            None,
        )
        .unwrap()
    }

    #[test]
    fn chunk_slice_1() {
        let chunks = (0..31)
            .map(|i| Chunk::new(i * 16, 16, [i]))
            .collect::<Vec<_>>();

        let ds = Dataset::new(
            Datatype::UInt(4),
            ByteOrder::BE,
            [31],
            chunks,
            [1],
            false,
            None,
        )
        .unwrap();

        ds.chunk_slices(None, None).for_each(drop);
        ds.chunk_slices(None, Some(&[4])).for_each(drop);
    }

    #[test]
    fn chunk_slice_11n() {
        let chunks = (0..2)
            .map(|i| (0..32).map(move |j| Chunk::new(i * 32 + j * 1000, 635000, [i, j, 0])))
            .flatten()
            .collect::<Vec<_>>();

        let ds = Dataset::new(
            Datatype::Int(2),
            ByteOrder::BE,
            [2, 32, 580],
            chunks,
            [1, 1, 580],
            false,
            None,
        )
        .unwrap();

        // ds.chunk_slices(None, Some(&[2, 4, 580, 1202]))
        //     .for_each(drop);
        ds.chunk_slices(None, Some(&[2, 32, 580])).for_each(drop);
    }

    #[test]
    fn chunk_slice_zero_count() {
        let d = test_dataset();
        assert_eq!(d.chunk_slices(None, Some(&[1, 0])).next(), None);
    }

    #[bench]
    fn chunk_slices_range(b: &mut Bencher) {
        let d = test_dataset();

        b.iter(|| d.chunk_slices(None, None).for_each(drop));
    }

    #[bench]
    fn make_chunk_slices_iterator(b: &mut Bencher) {
        let d = test_dataset();

        b.iter(|| test::black_box(d.chunk_slices(None, None)))
    }

    #[bench]
    fn chunk_at_coord(b: &mut Bencher) {
        let d = test_dataset();

        println!("chunks: {:#?}", d.chunks);

        assert_eq!(d.chunk_at_coord(&[0, 0]).offset, [ULE::new(0), ULE::new(0)]);
        assert_eq!(d.chunk_at_coord(&[0, 5]).offset, [ULE::new(0), ULE::new(0)]);
        assert_eq!(d.chunk_at_coord(&[5, 5]).offset, [ULE::new(0), ULE::new(0)]);
        assert_eq!(
            d.chunk_at_coord(&[0, 10]).offset,
            [ULE::new(0), ULE::new(10)]
        );
        assert_eq!(
            d.chunk_at_coord(&[0, 15]).offset,
            [ULE::new(0), ULE::new(10)]
        );
        assert_eq!(
            d.chunk_at_coord(&[10, 0]).offset,
            [ULE::new(10), ULE::new(0)]
        );
        assert_eq!(
            d.chunk_at_coord(&[10, 1]).offset,
            [ULE::new(10), ULE::new(0)]
        );
        assert_eq!(
            d.chunk_at_coord(&[15, 1]).offset,
            [ULE::new(10), ULE::new(0)]
        );

        b.iter(|| test::black_box(d.chunk_at_coord(&[15, 1])))
    }

    #[test]
    fn chunk_slices_scenarios() {
        let d = test_dataset();

        assert_eq!(
            d.chunk_slices(Some(&[0, 0]), Some(&[1, 10]))
                .collect::<Vec<(&Chunk<2>, u64, u64)>>(),
            [(&d.chunks[0], 0, 10)]
        );

        assert_eq!(
            d.chunk_slices(Some(&[0, 0]), Some(&[1, 20]))
                .collect::<Vec<(&Chunk<2>, u64, u64)>>(),
            [(&d.chunks[0], 0, 10), (&d.chunks[1], 0, 10)]
        );

        assert_eq!(
            d.chunk_slices(Some(&[0, 5]), Some(&[1, 15]))
                .collect::<Vec<(&Chunk<2>, u64, u64)>>(),
            [(&d.chunks[0], 5, 10), (&d.chunks[1], 0, 10)]
        );

        assert_eq!(
            d.chunk_slices(Some(&[0, 0]), Some(&[2, 10]))
                .collect::<Vec<(&Chunk<2>, u64, u64)>>(),
            [(&d.chunks[0], 0, 20)]
        );

        assert_eq!(
            d.chunk_slices(Some(&[0, 5]), Some(&[2, 10]))
                .collect::<Vec<(&Chunk<2>, u64, u64)>>(),
            [
                (&d.chunks[0], 5, 10),
                (&d.chunks[1], 0, 5),
                (&d.chunks[0], 15, 20),
                (&d.chunks[1], 10, 15)
            ]
        );

        assert_eq!(
            d.chunk_slices(Some(&[0, 0]), Some(&[2, 20]))
                .collect::<Vec<(&Chunk<2>, u64, u64)>>(),
            [
                (&d.chunks[0], 0, 10),
                (&d.chunks[1], 0, 10),
                (&d.chunks[0], 10, 20),
                (&d.chunks[1], 10, 20)
            ]
        );

        assert_eq!(
            d.chunk_slices(Some(&[2, 0]), Some(&[1, 10]))
                .collect::<Vec<(&Chunk<2>, u64, u64)>>(),
            [(&d.chunks[0], 20, 30),]
        );

        assert_eq!(
            d.chunk_slices(Some(&[2, 5]), Some(&[1, 10]))
                .collect::<Vec<(&Chunk<2>, u64, u64)>>(),
            [(&d.chunks[0], 25, 30), (&d.chunks[1], 20, 25),]
        );

        // column
        assert_eq!(
            d.chunk_slices(Some(&[2, 5]), Some(&[4, 1]))
                .collect::<Vec<(&Chunk<2>, u64, u64)>>(),
            [
                (&d.chunks[0], 25, 26),
                (&d.chunks[0], 35, 36),
                (&d.chunks[0], 45, 46),
                (&d.chunks[0], 55, 56),
            ]
        );

        assert_eq!(
            d.chunk_slices(Some(&[2, 15]), Some(&[4, 1]))
                .collect::<Vec<(&Chunk<2>, u64, u64)>>(),
            [
                (&d.chunks[1], 25, 26),
                (&d.chunks[1], 35, 36),
                (&d.chunks[1], 45, 46),
                (&d.chunks[1], 55, 56),
            ]
        );
    }

    #[test]
    fn coads_slice_all() {
        fn make_u64(u: u64) -> ULE {
            ULE::new(u)
        }

        let slices = vec![
            (
                Chunk {
                    addr: make_u64(31749),
                    size: make_u64(64800),
                    offset: [make_u64(0), make_u64(0), make_u64(0)],
                },
                0,
                16200,
            ),
            (
                Chunk {
                    addr: make_u64(96549),
                    size: make_u64(64800),
                    offset: [make_u64(1), make_u64(0), make_u64(0)],
                },
                0,
                16200,
            ),
            (
                Chunk {
                    addr: make_u64(161349),
                    size: make_u64(64800),
                    offset: [make_u64(2), make_u64(0), make_u64(0)],
                },
                0,
                16200,
            ),
            (
                Chunk {
                    addr: make_u64(226149),
                    size: make_u64(64800),
                    offset: [make_u64(3), make_u64(0), make_u64(0)],
                },
                0,
                16200,
            ),
            (
                Chunk {
                    addr: make_u64(290949),
                    size: make_u64(64800),
                    offset: [make_u64(4), make_u64(0), make_u64(0)],
                },
                0,
                16200,
            ),
            (
                Chunk {
                    addr: make_u64(355749),
                    size: make_u64(64800),
                    offset: [make_u64(5), make_u64(0), make_u64(0)],
                },
                0,
                16200,
            ),
            (
                Chunk {
                    addr: make_u64(420549),
                    size: make_u64(64800),
                    offset: [make_u64(6), make_u64(0), make_u64(0)],
                },
                0,
                16200,
            ),
            (
                Chunk {
                    addr: make_u64(485349),
                    size: make_u64(64800),
                    offset: [make_u64(7), make_u64(0), make_u64(0)],
                },
                0,
                16200,
            ),
            (
                Chunk {
                    addr: make_u64(550149),
                    size: make_u64(64800),
                    offset: [make_u64(8), make_u64(0), make_u64(0)],
                },
                0,
                16200,
            ),
            (
                Chunk {
                    addr: make_u64(614949),
                    size: make_u64(64800),
                    offset: [make_u64(9), make_u64(0), make_u64(0)],
                },
                0,
                16200,
            ),
            (
                Chunk {
                    addr: make_u64(679749),
                    size: make_u64(64800),
                    offset: [make_u64(10), make_u64(0), make_u64(0)],
                },
                0,
                16200,
            ),
            (
                Chunk {
                    addr: make_u64(744549),
                    size: make_u64(64800),
                    offset: [make_u64(11), make_u64(0), make_u64(0)],
                },
                0,
                16200,
            ),
        ];
        let slicebr = slices
            .iter()
            .map(|(c, s, e)| (c, *s, *e))
            .collect::<Vec<_>>();

        use crate::idx::Index;
        let i = Index::index("tests/data/coads_climatology.nc4").unwrap();
        let d = i.dataset("SST").unwrap();
        if let DatasetD::D3(d) = d {
            let sliced = d.chunk_slices(None, None).collect::<Vec<_>>();
            println!("slices: {:#?}", sliced);

            assert_eq!(sliced, slicebr);
        } else {
            panic!("wrong dims")
        }
    }

    #[test]
    fn serialize_variant_d2() {
        use flexbuffers::FlexbufferSerializer as ser;
        let d = DatasetD::D2(test_dataset());

        println!("serialize");
        let mut s = ser::new();
        d.serialize(&mut s).unwrap();

        println!("deserialize");
        let r = flexbuffers::Reader::get_root(s.view()).unwrap();
        let md = DatasetD::deserialize(r).unwrap();
        if let DatasetD::D2(md) = md {
            if let DatasetD::D2(d) = d {
                for (a, b) in izip!(d.chunk_shape.iter(), md.chunk_shape.iter()) {
                    assert_eq!(a, b);
                }
            } else {
                panic!("wrong variant");
            }
        } else {
            panic!("wrong variant");
        }
    }
}
