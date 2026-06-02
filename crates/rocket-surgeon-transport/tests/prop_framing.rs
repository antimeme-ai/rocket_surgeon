//! Property-based tests for Content-Length framing (`read_message` /
//! `write_message`) and the stdio transport that layers JSON-RPC on top.
//!
//! MATERIA oracle tiers:
//!   - Tier 4 (roundtrip): `write_message` then `read_message` is the identity
//!     on arbitrary UTF-8 bodies — including bodies that *contain* framing
//!     syntax (`\r\n\r\n`, `Content-Length:`), which a naive re-parser would
//!     corrupt.
//!   - Tier 6 (model): a stream of N framed messages reads back as exactly the
//!     N bodies in FIFO order (model = the input `Vec`).
//!   - Tier 4 (metamorphic): unrelated extra headers and the case of the
//!     `Content-Length` key do not change the decoded body.
//!   - Exception-raising: missing / non-numeric / over-limit Content-Length,
//!     EOF, and truncated bodies each produce the specific `FramingError`,
//!     never a panic and never a wrong body.

use proptest::prelude::*;
use proptest::strategy::ValueTree;
use proptest::test_runner::{Config as PtConfig, TestRunner};
use rocket_surgeon_protocol::jsonrpc::{Request, RequestId};
use rocket_surgeon_transport::Transport;
use rocket_surgeon_transport::framing::{
    FramingError, MAX_MESSAGE_BYTES, read_message, write_message,
};
use rocket_surgeon_transport::stdio::StdioTransport;
use std::io::Cursor;

// ---------------------------------------------------------------------------
// Body strategy: ordinary strings plus adversarial bodies that embed framing
// syntax, unicode, and control characters.
// ---------------------------------------------------------------------------

fn adversarial_body() -> impl Strategy<Value = String> {
    prop::collection::vec(
        prop_oneof![
            Just("\r\n".to_owned()),
            Just("\r\n\r\n".to_owned()),
            Just("Content-Length: 7".to_owned()),
            Just("🚀héllo".to_owned()),
            Just("\0\t".to_owned()),
            "[ -~]{0,12}".prop_map(|s| s),
        ],
        0..12,
    )
    .prop_map(|parts| parts.concat())
}

fn body() -> impl Strategy<Value = String> {
    prop_oneof![
        3 => any::<String>(),
        3 => adversarial_body(),
        1 => Just(String::new()),
    ]
}

// ---------------------------------------------------------------------------
// Generator distribution evidence
// ---------------------------------------------------------------------------

#[test]
fn body_generator_distribution() {
    const N: u32 = 3000;
    let mut runner = TestRunner::new(PtConfig::default());
    let strat = body();
    let (mut empty, mut has_crlf, mut has_cl, mut non_ascii, mut plain) = (0u32, 0, 0, 0, 0);
    for _ in 0..N {
        let b = strat.new_tree(&mut runner).unwrap().current();
        if b.is_empty() {
            empty += 1;
        }
        if b.contains("\r\n") {
            has_crlf += 1;
        }
        if b.to_ascii_lowercase().contains("content-length") {
            has_cl += 1;
        }
        if !b.is_ascii() {
            non_ascii += 1;
        }
        if b.is_ascii() && !b.contains("\r\n") && !b.is_empty() {
            plain += 1;
        }
    }
    eprintln!(
        "body distribution over {N}: empty={empty} has_crlf={has_crlf} \
         has_content_length={has_cl} non_ascii={non_ascii} plain={plain}"
    );
    assert!(empty > 50, "empty bodies underrepresented: {empty}");
    assert!(
        has_crlf > 100,
        "framing-syntax bodies underrepresented: {has_crlf}"
    );
    assert!(
        non_ascii > 50,
        "unicode bodies underrepresented: {non_ascii}"
    );
    assert!(plain > 100, "plain bodies underrepresented: {plain}");
}

// ---------------------------------------------------------------------------
// Roundtrip + multi-message model
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig { cases: 400, ..ProptestConfig::default() })]

    /// write_message ; read_message == identity, even when the body itself
    /// contains `\r\n\r\n` or a `Content-Length:` line.
    #[test]
    fn write_read_roundtrip(b in body()) {
        let mut wire = Vec::new();
        write_message(&mut wire, &b).unwrap();
        let mut cursor = Cursor::new(wire);
        let got = read_message(&mut cursor).unwrap();
        prop_assert_eq!(got, b);
    }

    /// A concatenation of N framed messages decodes to exactly those N bodies
    /// in order (model = the input Vec).
    #[test]
    fn multi_message_fifo(bodies in prop::collection::vec(body(), 0..8)) {
        let mut wire = Vec::new();
        for b in &bodies {
            write_message(&mut wire, b).unwrap();
        }
        let mut cursor = Cursor::new(wire);
        let mut decoded = Vec::new();
        for _ in 0..bodies.len() {
            decoded.push(read_message(&mut cursor).unwrap());
        }
        prop_assert_eq!(decoded, bodies);
        // The stream is now exhausted: the next read must be a clean EOF error.
        prop_assert!(matches!(read_message(&mut cursor), Err(FramingError::Io(_))));
    }
}

// ---------------------------------------------------------------------------
// Metamorphic relations
// ---------------------------------------------------------------------------

/// Header keys that are guaranteed NOT to collide with `content-length`.
fn benign_header_line() -> impl Strategy<Value = String> {
    ("[A-Za-z][A-Za-z0-9-]{0,10}", "[ -~&&[^\r\n]]{0,20}")
        .prop_filter("key must not be content-length", |(k, _)| {
            !k.eq_ignore_ascii_case("content-length")
        })
        .prop_map(|(k, v)| format!("X-{k}: {v}\r\n"))
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 300, ..ProptestConfig::default() })]

    /// Inserting arbitrary unrelated headers (before and after Content-Length)
    /// does not change the decoded body.
    #[test]
    fn extra_headers_are_invariant(
        b in "[ -~]{0,64}",
        pre in prop::collection::vec(benign_header_line(), 0..4),
        post in prop::collection::vec(benign_header_line(), 0..4),
    ) {
        let mut wire = String::new();
        for h in &pre {
            wire.push_str(h);
        }
        wire.push_str("Content-Length: ");
        wire.push_str(&b.len().to_string());
        wire.push_str("\r\n");
        for h in &post {
            wire.push_str(h);
        }
        wire.push_str("\r\n");
        wire.push_str(&b);

        let mut cursor = Cursor::new(wire.into_bytes());
        let got = read_message(&mut cursor).unwrap();
        prop_assert_eq!(got, b);
    }

    /// The Content-Length key is matched case-insensitively: any casing decodes
    /// to the same body.
    #[test]
    fn content_length_case_insensitive(b in "[ -~]{0,64}", mask in any::<u64>()) {
        let key: String = "Content-Length"
            .chars()
            .enumerate()
            .map(|(i, c)| if (mask >> (i % 64)) & 1 == 1 { c.to_ascii_uppercase() } else { c.to_ascii_lowercase() })
            .collect();
        let wire = format!("{key}: {}\r\n\r\n{b}", b.len());
        let mut cursor = Cursor::new(wire.into_bytes());
        let got = read_message(&mut cursor).unwrap();
        prop_assert_eq!(got, b);
    }
}

// ---------------------------------------------------------------------------
// Exception-raising properties
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig { cases: 300, ..ProptestConfig::default() })]

    /// Headers present but no Content-Length => MissingContentLength.
    #[test]
    fn missing_content_length(
        headers in prop::collection::vec(benign_header_line(), 0..5),
        trailer in "[ -~]{0,16}",
    ) {
        let mut wire = String::new();
        for h in &headers {
            wire.push_str(h);
        }
        wire.push_str("\r\n");
        wire.push_str(&trailer);
        let mut cursor = Cursor::new(wire.into_bytes());
        prop_assert!(matches!(
            read_message(&mut cursor),
            Err(FramingError::MissingContentLength)
        ));
    }

    /// A non-numeric Content-Length value => InvalidContentLength.
    #[test]
    fn invalid_content_length(junk in "[A-Za-z!@#-][A-Za-z0-9 ]{0,8}") {
        // `junk` always starts with a non-digit, non-sign char so usize::parse fails.
        let wire = format!("Content-Length: {junk}\r\n\r\nxxxx");
        let mut cursor = Cursor::new(wire.into_bytes());
        prop_assert!(matches!(
            read_message(&mut cursor),
            Err(FramingError::InvalidContentLength)
        ));
    }

    /// A Content-Length beyond the cap => MessageTooLarge, checked BEFORE any
    /// allocation (so we can name an absurd size without OOM risk).
    #[test]
    fn message_too_large(extra in 1u64..=u64::from(u32::MAX)) {
        let huge = MAX_MESSAGE_BYTES as u64 + extra;
        let wire = format!("Content-Length: {huge}\r\n\r\n");
        let mut cursor = Cursor::new(wire.into_bytes());
        prop_assert!(matches!(
            read_message(&mut cursor),
            Err(FramingError::MessageTooLarge(n)) if n as u64 == huge
        ));
    }

    /// A correct header but a body shorter than promised => an Io error
    /// (UnexpectedEof from read_exact), never a short/garbage body.
    #[test]
    fn truncated_body_errors(b in "[ -~]{1,64}", drop_n in 1usize..=64) {
        let drop_n = drop_n.min(b.len());
        let full = format!("Content-Length: {}\r\n\r\n{}", b.len(), b);
        let mut bytes = full.into_bytes();
        bytes.truncate(bytes.len() - drop_n);
        let mut cursor = Cursor::new(bytes);
        prop_assert!(matches!(
            read_message(&mut cursor),
            Err(FramingError::Io(_))
        ));
    }
}

// ---------------------------------------------------------------------------
// Integration: StdioTransport request roundtrip (framing + serde + transport)
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig { cases: 300, ..ProptestConfig::default() })]

    /// A serde-serialized, Content-Length-framed Request is received through the
    /// transport with its id and method intact — the framing + serde + transport
    /// recv stack is an identity on the wire-observable fields.
    #[test]
    fn transport_request_roundtrip(id in any::<i64>(), method in "[a-zA-Z/$._-]{1,24}") {
        let req = Request::new(RequestId::Number(id), &method, serde_json::Value::Null);

        // Frame exactly as `send_request` would (serde_json::to_string + write_message).
        let mut wire = Vec::new();
        write_message(&mut wire, &serde_json::to_string(&req).unwrap()).unwrap();

        let mut receiver = StdioTransport::new(Cursor::new(wire), Vec::<u8>::new());
        let got = receiver.recv_request().unwrap();
        prop_assert_eq!(got.id, RequestId::Number(id));
        prop_assert_eq!(got.method, method);
    }
}
