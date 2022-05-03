#![feature(test)]
extern crate test;
use futures::executor::block_on_stream;
use futures::{pin_mut, Stream, StreamExt};
use hidefix::prelude::*;
use test::Bencher;

fn consume_stream<S: Stream>(rt: &mut tokio::runtime::Runtime, s: S) {
    rt.block_on(async move {
        pin_mut!(s);
        while let Some(_) = s.next().await {}
    });
}

#[bench]
fn chunked_1d_values(b: &mut Bencher) {
    let mut rt = tokio::runtime::Runtime::new().unwrap();
    let i = Index::index("tests/data/dmrpp/chunked_oneD.h5").unwrap();
    let r = i.streamer("d_4_chunks").unwrap();

    b.iter(|| {
        let v = r.stream_values::<f32>(None, None);
        consume_stream(&mut rt, v);
    })
}

#[bench]
fn gzip_shuffle_2d_bytes(b: &mut Bencher) {
    let mut rt = tokio::runtime::Runtime::new().unwrap();
    let i = Index::index("tests/data/gzip_shuffle_2d.h5").unwrap();
    let r = i.streamer("data").unwrap();

    b.iter(|| {
        let v = r.stream(None, None);
        consume_stream(&mut rt, v);
    })
}

#[bench]
fn coads_values(b: &mut Bencher) {
    let mut rt = tokio::runtime::Runtime::new().unwrap();
    let i = Index::index("tests/data/coads_climatology.nc4").unwrap();
    let r = i.streamer("SST").unwrap();

    {
        let v = r.stream_values::<f32>(None, None);
        let vs: Vec<f32> = block_on_stream(v).flatten().flatten().collect();

        let h = hdf5::File::open("tests/data/coads_climatology.nc4").unwrap();
        let d = h.dataset("SST").unwrap();
        assert_eq!(d.read_raw::<f32>().unwrap(), vs);
    }

    b.iter(|| {
        let v = r.stream_values::<f32>(None, None);
        consume_stream(&mut rt, v);
    })
}

#[bench]
fn coads_bytes(b: &mut Bencher) {
    let mut rt = tokio::runtime::Runtime::new().unwrap();
    let i = Index::index("tests/data/coads_climatology.nc4").unwrap();
    let r = i.streamer("SST").unwrap();

    b.iter(|| {
        let v = r.stream(None, None);
        consume_stream(&mut rt, v);
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
            let mut r = v.into_async_read();
            let mut buf = Vec::with_capacity(8 * 1024);
            r.read_to_end(&mut buf).await.unwrap();
        })
    })
}
