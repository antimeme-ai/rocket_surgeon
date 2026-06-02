//! Property tests for the pure (no-shm) layer: `RingConfig` validation and
//! geometry, and the `serialize_probe_frame` wire encoder.
//!
//! MATERIA oracle tiers:
//!   - Tier 6 (model): `RingConfig::new` agrees with an independent reference
//!     predicate over arbitrary (backuptics, `slot_size`), including the full
//!     u64 range (overflow path) — no shm regions are created here, so we can
//!     range over extreme values cheaply.
//!   - Tier 7 (spec): every field written by `serialize_probe_frame` round-trips
//!     from its documented byte offset with little-endian encoding. The offsets
//!     ARE the spec (they must match the Python side), so this is the strongest
//!     oracle available.
//!   - Exception-raising: non-power-of-two counts, undersized slots, and
//!     overflowing geometry each yield the specific typed error.

use proptest::prelude::*;
use rocket_surgeon_shm::{
    CONTROL_SIZE, FRAME_OFFSET_COMP_ID, FRAME_OFFSET_DATA_OFF, FRAME_OFFSET_DTYPE,
    FRAME_OFFSET_FLAGS, FRAME_OFFSET_GENERATION, FRAME_OFFSET_LAYER, FRAME_OFFSET_NDIM,
    FRAME_OFFSET_RANK, FRAME_OFFSET_SHAPE, FRAME_OFFSET_SIZE, FRAME_OFFSET_TICK_ID,
    PROBE_FRAME_HEADER_SIZE, RingConfig, ShmError, serialize_probe_frame,
};

// ---------------------------------------------------------------------------
// RingConfig::new — model oracle + exception-raising
// ---------------------------------------------------------------------------

/// Independent reference for whether `new` should succeed, mirroring the spec
/// (power-of-two count, `slot_size` >= header, no geometry overflow) WITHOUT
/// reusing the implementation's control flow.
#[derive(Debug, PartialEq, Eq)]
enum Expect {
    NotPowerOfTwo,
    InvalidConfig,
    Ok,
}

fn reference_expect(backuptics: u32, slot_size: u64) -> Expect {
    let is_pow2 = backuptics != 0 && backuptics.is_power_of_two();
    if !is_pow2 {
        return Expect::NotPowerOfTwo;
    }
    if slot_size < PROBE_FRAME_HEADER_SIZE as u64 {
        return Expect::InvalidConfig;
    }
    match u64::from(backuptics)
        .checked_mul(slot_size)
        .and_then(|v| v.checked_add(CONTROL_SIZE as u64))
    {
        None => Expect::InvalidConfig,
        Some(_) => Expect::Ok,
    }
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 2000, ..ProptestConfig::default() })]

    /// `RingConfig::new` matches the reference classifier on every input,
    /// across the entire u32 x u64 space (powers of two are rare in a uniform
    /// u32 draw, so we also inject them explicitly below).
    #[test]
    fn new_matches_reference(backuptics in any::<u32>(), slot_size in any::<u64>()) {
        let got = RingConfig::new(backuptics, slot_size);
        match (reference_expect(backuptics, slot_size), got) {
            (Expect::Ok, Ok(cfg)) => {
                prop_assert_eq!(cfg.backuptics, backuptics);
                prop_assert_eq!(cfg.slot_size, slot_size);
            }
            (Expect::NotPowerOfTwo, Err(ShmError::NotPowerOfTwo(b))) => {
                prop_assert_eq!(b, backuptics);
            }
            (Expect::InvalidConfig, Err(ShmError::InvalidConfig(_))) => {}
            (exp, got) => prop_assert!(false, "mismatch: reference={:?} got={:?}", exp, got),
        }
    }

    /// Powers of two with a valid slot_size always construct successfully, and
    /// the geometry helpers obey their algebraic spec.
    #[test]
    fn power_of_two_configs_have_consistent_geometry(
        exp in 0u32..=20,                // backuptics = 2^exp, up to ~1M
        slot_size in (PROBE_FRAME_HEADER_SIZE as u64)..=(1u64 << 32),
    ) {
        let backuptics = 1u32 << exp;
        let cfg = RingConfig::new(backuptics, slot_size).unwrap();

        prop_assert_eq!(cfg.mask(), u64::from(backuptics) - 1);
        prop_assert_eq!(cfg.slot_data_capacity(), slot_size as usize - PROBE_FRAME_HEADER_SIZE);
        prop_assert_eq!(
            cfg.region_size(),
            CONTROL_SIZE + backuptics as usize * slot_size as usize
        );
    }
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 1000, ..ProptestConfig::default() })]

    /// slot_offset is the model `CONTROL + (tick mod backuptics) * slot_size`,
    /// and it is periodic with period = backuptics (metamorphic relation),
    /// proving the bitmask wrap matches modular arithmetic for powers of two.
    #[test]
    fn slot_offset_is_modular_and_periodic(
        exp in 0u32..=16,
        slot_size in (PROBE_FRAME_HEADER_SIZE as u64)..=65536,
        tick in 0u64..(1u64 << 40),
    ) {
        let backuptics = 1u32 << exp;
        let cfg = RingConfig::new(backuptics, slot_size).unwrap();

        let expected = CONTROL_SIZE + ((tick & cfg.mask()) as usize) * slot_size as usize;
        prop_assert_eq!(cfg.slot_offset(tick), expected);

        // Modular wrap == bitmask wrap.
        prop_assert_eq!(tick & cfg.mask(), tick % u64::from(backuptics));

        // Period = backuptics (no u64 wrap because tick < 2^40 and bt <= 2^16).
        prop_assert_eq!(
            cfg.slot_offset(tick),
            cfg.slot_offset(tick + u64::from(backuptics))
        );

        // The whole slot stays inside the region.
        prop_assert!(cfg.slot_offset(tick) + slot_size as usize <= cfg.region_size());
    }
}

// ---------------------------------------------------------------------------
// serialize_probe_frame — spec-oracle roundtrip
// ---------------------------------------------------------------------------

fn le_u16(buf: &[u8], off: usize) -> u16 {
    u16::from_le_bytes(buf[off..off + 2].try_into().unwrap())
}
fn le_u32(buf: &[u8], off: usize) -> u32 {
    u32::from_le_bytes(buf[off..off + 4].try_into().unwrap())
}
fn le_u64(buf: &[u8], off: usize) -> u64 {
    u64::from_le_bytes(buf[off..off + 8].try_into().unwrap())
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 1000, ..ProptestConfig::default() })]

    /// Decode every field back from its documented offset; each must equal the
    /// value that was encoded. This pins the on-wire layout the Python probe
    /// side depends on.
    #[test]
    fn probe_frame_fields_roundtrip(
        rank in any::<u32>(),
        layer in any::<u32>(),
        comp_id in any::<u16>(),
        dtype in any::<u8>(),
        ndim in any::<u8>(),
        shape in prop::array::uniform8(any::<u32>()),
        tick_id in any::<u64>(),
        data_off in any::<u64>(),
        size in any::<u64>(),
        flags in any::<u32>(),
        generation in any::<u32>(),
    ) {
        let buf = serialize_probe_frame(
            rank, layer, comp_id, dtype, ndim, &shape, tick_id, data_off, size, flags, generation,
        );
        prop_assert_eq!(buf.len(), PROBE_FRAME_HEADER_SIZE);

        prop_assert_eq!(le_u32(&buf, FRAME_OFFSET_RANK), rank);
        prop_assert_eq!(le_u32(&buf, FRAME_OFFSET_LAYER), layer);
        prop_assert_eq!(le_u16(&buf, FRAME_OFFSET_COMP_ID), comp_id);
        prop_assert_eq!(buf[FRAME_OFFSET_DTYPE], dtype);
        prop_assert_eq!(buf[FRAME_OFFSET_NDIM], ndim);
        for (i, &dim) in shape.iter().enumerate() {
            prop_assert_eq!(le_u32(&buf, FRAME_OFFSET_SHAPE + i * 4), dim);
        }
        prop_assert_eq!(le_u64(&buf, FRAME_OFFSET_TICK_ID), tick_id);
        prop_assert_eq!(le_u64(&buf, FRAME_OFFSET_DATA_OFF), data_off);
        prop_assert_eq!(le_u64(&buf, FRAME_OFFSET_SIZE), size);
        prop_assert_eq!(le_u32(&buf, FRAME_OFFSET_FLAGS), flags);
        prop_assert_eq!(le_u32(&buf, FRAME_OFFSET_GENERATION), generation);
    }

    /// Distinct fields occupy disjoint byte ranges: setting exactly one field
    /// (all others zero) must leave a buffer that differs from all-zero ONLY in
    /// that field's bytes. Catches overlapping-offset regressions that a plain
    /// roundtrip can miss when two fields happen to share a value.
    #[test]
    fn setting_size_touches_only_size_bytes(size in 1u64..=u64::MAX) {
        let buf = serialize_probe_frame(0, 0, 0, 0, 0, &[0u32; 8], 0, 0, size, 0, 0);
        for (i, &b) in buf.iter().enumerate() {
            let in_size_field = (FRAME_OFFSET_SIZE..FRAME_OFFSET_SIZE + 8).contains(&i);
            if !in_size_field {
                prop_assert_eq!(b, 0, "byte {} outside size field was mutated", i);
            }
        }
        prop_assert_eq!(le_u64(&buf, FRAME_OFFSET_SIZE), size);
    }
}
