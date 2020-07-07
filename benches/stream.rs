#![feature(test)]
extern crate test;
use test::Bencher;
use futures::executor::block_on_stream;
use futures::pin_mut;
use hidefix::idx::Index;

#[bench]
fn chunked_1d_values(b: &mut Bencher) {
    let i = Index::index("tests/data/dmrpp/chunked_oneD.h5").unwrap();
    let r = i.streamer("d_4_chunks").unwrap();

    b.iter(|| {
        let v = r.stream_values::<f32>(None, None);
        pin_mut!(v);
        block_on_stream(v).for_each(drop);
    })
}

#[bench]
fn gzip_shuffle_2d_bytes(b: &mut Bencher) {
    let i = Index::index("tests/data/gzip_shuffle_2d.h5").unwrap();
    let r = i.streamer("data").unwrap();

    b.iter(|| {
        let v = r.stream(None, None);
        pin_mut!(v);
        block_on_stream(v).for_each(drop);
    })
}

#[bench]
fn coads_values(b: &mut Bencher) {
    let i = Index::index("tests/data/coads_climatology.nc4").unwrap();
    let r = i.streamer("SST").unwrap();

    {
        let v = r.stream_values::<f32>(None, None);
        pin_mut!(v);
        let vs: Vec<f32> = block_on_stream(v).flatten().flatten().collect();

        let h = hdf5::File::open("tests/data/coads_climatology.nc4").unwrap();
        let d = h.dataset("SST").unwrap();
        assert_eq!(d.read_raw::<f32>().unwrap(), vs);
    }

    b.iter(|| {
        let v = r.stream_values::<f32>(None, None);
        pin_mut!(v);
        block_on_stream(v).for_each(drop);
    })
}

#[bench]
fn coads_bytes(b: &mut Bencher) {
    let i = Index::index("tests/data/coads_climatology.nc4").unwrap();
    let r = i.streamer("SST").unwrap();

    b.iter(|| {
        let v = r.stream(None, None);
        pin_mut!(v);
        block_on_stream(v).for_each(drop);
    })
}

#[bench]
fn coads_async_read(b: &mut Bencher) {
    use futures::executor::block_on;
    use futures::io::AsyncReadExt;
    use futures::stream::TryStreamExt;

    let i = Index::index("tests/data/coads_climatology.nc4").unwrap();
    let r = i.streamer("SST").unwrap();

    b.iter(|| {
        block_on(async {
            let v = r
                .stream(None, None)
                .map_err(|_| std::io::ErrorKind::UnexpectedEof.into());
            pin_mut!(v);
            let mut r = v.into_async_read();
            let mut buf = Vec::with_capacity(8 * 1024);
            r.read_to_end(&mut buf).await.unwrap();
        })
    })
}

