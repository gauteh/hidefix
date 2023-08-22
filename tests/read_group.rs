use hdf5::File;
use hidefix::idx::Index;

#[test]
fn read_group() {
    let atl11 = Index::index("tests/data/ATL11_078805_0304_02_v002.h5").unwrap();
    let hf5 = File::open(atl11.path().unwrap()).unwrap();
    let group = hf5.group("/pt2/corrected_h").unwrap();
    let members = group.member_names().unwrap();

    assert_eq!(
        members,
        [
            "cycle_number",
            "delta_time",
            "h_corr",
            "h_corr_sigma",
            "h_corr_sigma_systematic",
            "latitude",
            "longitude",
            "quality_summary",
            "ref_pt"
        ]
    );
}
