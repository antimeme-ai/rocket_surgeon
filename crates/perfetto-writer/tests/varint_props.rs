//! Property + differential tests for LEB128 varint encoding.
//!
//! Oracle hierarchy:
//!   - tier 6 (model): our `encode_varint` must agree byte-for-byte with prost's
//!     reference encoder, and prost's decoder must recover the original value.
//!   - tier 5 (property): continuation-bit and length-formula invariants hold for
//!     every u64.
//!   - tier 4 (roundtrip): a fully independent local decoder recovers the value
//!     consuming exactly the produced bytes (no trailing slack).
//!   - tier 7 (specification): explicit boundary table with hand-derived lengths.

use perfetto_writer::varint::{encode_varint, field1_tag_and_length};
use proptest::prelude::*;
use proptest::strategy::ValueTree;
use proptest::test_runner::TestRunner;

/// Independent reference decoder — the roundtrip oracle. Returns the decoded
/// value and number of bytes consumed, or `None` if the buffer is a truncated /
/// overlong varint. Deliberately not the production decoder (there is none);
/// disagreement with `encode_varint` is a real finding.
fn decode_ref(buf: &[u8]) -> Option<(u64, usize)> {
    let mut value: u64 = 0;
    let mut shift: u32 = 0;
    for (i, &byte) in buf.iter().enumerate() {
        if shift >= 64 {
            return None; // overlong: more than 10 groups
        }
        value |= u64::from(byte & 0x7F) << shift;
        if byte & 0x80 == 0 {
            return Some((value, i + 1));
        }
        shift += 7;
    }
    None // truncated: ran out of bytes with continuation bit set
}

/// The full boundary set the brief calls for: every 7-bit group transition, ±1,
/// plus the extremes. Each entry is `(value, expected_encoded_len)`.
fn boundary_table() -> Vec<(u64, usize)> {
    let mut out = vec![(0u64, 1usize), (1, 1)];
    // 2^(7k) - 1 is the largest value encodable in k bytes; 2^(7k) needs k+1.
    for k in 1..=9u32 {
        let threshold = 1u64 << (7 * k); // first value needing k+1 bytes
        out.push((threshold - 1, k as usize)); // max in k bytes
        out.push((threshold, k as usize + 1)); // min in k+1 bytes
        out.push((threshold + 1, k as usize + 1));
    }
    out.push((u64::MAX, 10));
    out.push((u64::MAX - 1, 10));
    out
}

/// Significant-bit-count length formula: ceil(bits/7), min 1.
fn expected_len(v: u64) -> usize {
    if v == 0 {
        return 1;
    }
    let bits = 64 - v.leading_zeros() as usize;
    bits.div_ceil(7)
}

/// Value generator with deliberate coverage across all 10 possible byte lengths.
/// `any::<u64>()` alone is ~uniform and would put >99% of mass in the 9–10 byte
/// buckets; shifting a uniform draw right by a uniform amount spreads the
/// significant-bit count evenly, and the boundary arm injects the exact corners.
fn varint_value() -> impl Strategy<Value = u64> {
    let boundaries: Vec<u64> = boundary_table().into_iter().map(|(v, _)| v).collect();
    prop_oneof![
        4 => (any::<u64>(), 0u32..64u32).prop_map(|(v, s)| v >> s),
        1 => prop::sample::select(boundaries),
    ]
}

proptest! {
    /// tier 4 — self roundtrip via the independent decoder, exact consumption.
    #[test]
    fn roundtrip_consumes_exactly(v in varint_value()) {
        let mut buf = Vec::new();
        encode_varint(v, &mut buf);
        let (decoded, consumed) = decode_ref(&buf).expect("well-formed varint");
        prop_assert_eq!(decoded, v);
        prop_assert_eq!(consumed, buf.len(), "no trailing bytes");
    }

    /// tier 6 — differential against prost's reference codec (model oracle).
    #[test]
    fn agrees_with_prost(v in varint_value()) {
        let mut ours = Vec::new();
        encode_varint(v, &mut ours);

        let mut theirs = Vec::new();
        prost::encoding::encode_varint(v, &mut theirs);
        prop_assert_eq!(&ours, &theirs, "byte-for-byte disagreement with prost");

        // prost decodes our bytes back to the original, consuming all of them.
        let mut slice: &[u8] = &ours;
        let round = prost::encoding::decode_varint(&mut slice).expect("prost decodes ours");
        prop_assert_eq!(round, v);
        prop_assert!(slice.is_empty(), "prost left trailing bytes");
    }

    /// tier 5 — LEB128 structural invariants.
    #[test]
    fn continuation_and_length_invariants(v in varint_value()) {
        let mut buf = Vec::new();
        encode_varint(v, &mut buf);

        prop_assert!(!buf.is_empty());
        prop_assert!(buf.len() <= 10, "u64 varint never exceeds 10 bytes");
        prop_assert_eq!(buf.len(), expected_len(v), "length matches bit-count formula");

        let (last, init) = buf.split_last().unwrap();
        for &b in init {
            prop_assert_eq!(b & 0x80, 0x80, "non-terminal byte must continue");
        }
        prop_assert_eq!(last & 0x80, 0, "terminal byte must not continue");
        // Minimal encoding: the terminal byte is never a spurious zero
        // (would mean a non-minimal/overlong form).
        if buf.len() > 1 {
            prop_assert_ne!(*last, 0, "no overlong trailing zero group");
        }
    }

    /// tier 6 — field-1 LEN frame: 0x0A tag (field 1, wire-type 2) then a length
    /// varint that prost reads back as the payload length.
    #[test]
    fn field1_frame_decodes(len in 0usize..1_000_000) {
        let mut buf = Vec::new();
        field1_tag_and_length(len, &mut buf);
        prop_assert_eq!(buf[0], 0x0A);
        // 0x0A = (field 1 << 3) | wire-type 2.
        prop_assert_eq!(buf[0] >> 3, 1, "field number 1");
        prop_assert_eq!(buf[0] & 0x07, 2, "wire-type LEN");

        let (decoded, consumed) = decode_ref(&buf[1..]).unwrap();
        prop_assert_eq!(decoded, len as u64);
        prop_assert_eq!(consumed + 1, buf.len());
    }
}

/// tier 7 — exhaustive, hand-derived boundary table.
#[test]
fn boundary_table_exact() {
    for (v, expect_len) in boundary_table() {
        let mut buf = Vec::new();
        encode_varint(v, &mut buf);
        assert_eq!(buf.len(), expect_len, "len for {v:#x}");
        let (decoded, consumed) = decode_ref(&buf).unwrap();
        assert_eq!(decoded, v, "roundtrip {v:#x}");
        assert_eq!(consumed, buf.len());

        // cross-check against prost too
        let mut slice: &[u8] = &buf;
        assert_eq!(prost::encoding::decode_varint(&mut slice).unwrap(), v);
    }
}

/// Generator-distribution evidence (MATERIA: "measure your generators"). Samples
/// the value strategy and buckets by encoded byte-length; asserts every length
/// 1..=10 is exercised, so the property tests above actually span the range.
#[test]
fn generator_covers_all_byte_lengths() {
    const N: usize = 20_000;
    let mut runner = TestRunner::deterministic();
    let strat = varint_value();
    let mut hist = [0u32; 11]; // index = byte length
    for _ in 0..N {
        let v = strat.new_tree(&mut runner).unwrap().current();
        let mut buf = Vec::new();
        encode_varint(v, &mut buf);
        hist[buf.len()] += 1;
    }
    eprintln!(
        "varint byte-length distribution over {N} samples: {:?}",
        &hist[1..=10]
    );
    for (len, &count) in hist.iter().enumerate().take(11).skip(1) {
        assert!(
            count > 0,
            "byte-length {len} never generated — generator too narrow"
        );
    }
}
