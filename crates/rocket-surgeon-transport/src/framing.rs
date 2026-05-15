use std::io::{self, BufRead, Write};

pub const MAX_MESSAGE_BYTES: usize = 16 * 1024 * 1024;

#[derive(Debug, thiserror::Error)]
pub enum FramingError {
    #[error("missing Content-Length header")]
    MissingContentLength,
    #[error("invalid Content-Length value")]
    InvalidContentLength,
    #[error("message too large ({0} bytes, max {MAX_MESSAGE_BYTES})")]
    MessageTooLarge(usize),
    #[error("{0}")]
    Io(#[from] io::Error),
}

pub fn read_message(reader: &mut impl BufRead) -> Result<String, FramingError> {
    let mut content_length: Option<usize> = None;

    loop {
        let mut line = String::new();
        let n = reader.read_line(&mut line)?;
        if n == 0 {
            return Err(FramingError::Io(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "EOF",
            )));
        }

        let trimmed = line.trim_end_matches(['\r', '\n']);

        if trimmed.is_empty() {
            break;
        }

        if let Some((key, value)) = trimmed.split_once(':') {
            if key.eq_ignore_ascii_case("content-length") {
                content_length = Some(
                    value
                        .trim()
                        .parse()
                        .map_err(|_| FramingError::InvalidContentLength)?,
                );
            }
        }
    }

    let content_length = content_length.ok_or(FramingError::MissingContentLength)?;

    if content_length > MAX_MESSAGE_BYTES {
        return Err(FramingError::MessageTooLarge(content_length));
    }

    let mut body = vec![0u8; content_length];
    reader.read_exact(&mut body)?;

    String::from_utf8(body)
        .map_err(|e| FramingError::Io(io::Error::new(io::ErrorKind::InvalidData, e)))
}

pub fn write_message(writer: &mut impl Write, body: &str) -> Result<(), FramingError> {
    let header = format!("Content-Length: {}\r\n\r\n", body.len());
    writer.write_all(header.as_bytes())?;
    writer.write_all(body.as_bytes())?;
    writer.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn frame(body: &str) -> Vec<u8> {
        format!("Content-Length: {}\r\n\r\n{}", body.len(), body).into_bytes()
    }

    #[test]
    fn round_trip() {
        let body = r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}"#;
        let framed = frame(body);
        let mut cursor = Cursor::new(framed);
        let result = read_message(&mut cursor).unwrap();
        assert_eq!(result, body);
    }

    #[test]
    fn write_then_read_round_trip() {
        let body = r#"{"test":true}"#;
        let mut buf = Vec::new();
        write_message(&mut buf, body).unwrap();
        let mut cursor = Cursor::new(buf);
        let result = read_message(&mut cursor).unwrap();
        assert_eq!(result, body);
    }

    #[test]
    fn multiple_messages() {
        let msg1 = r#"{"id":1}"#;
        let msg2 = r#"{"id":2}"#;
        let mut buf = Vec::new();
        write_message(&mut buf, msg1).unwrap();
        write_message(&mut buf, msg2).unwrap();

        let mut cursor = Cursor::new(buf);
        assert_eq!(read_message(&mut cursor).unwrap(), msg1);
        assert_eq!(read_message(&mut cursor).unwrap(), msg2);
    }

    #[test]
    fn missing_content_length() {
        let mut cursor = Cursor::new(b"Bad-Header: 10\r\n\r\nxxxxxxxxxx".to_vec());
        let err = read_message(&mut cursor).unwrap_err();
        assert!(matches!(err, FramingError::MissingContentLength));
    }

    #[test]
    fn invalid_content_length_value() {
        let mut cursor = Cursor::new(b"Content-Length: abc\r\n\r\n".to_vec());
        let err = read_message(&mut cursor).unwrap_err();
        assert!(matches!(err, FramingError::InvalidContentLength));
    }

    #[test]
    fn eof_returns_error() {
        let mut cursor = Cursor::new(Vec::new());
        let err = read_message(&mut cursor).unwrap_err();
        assert!(matches!(err, FramingError::Io(_)));
    }

    #[test]
    fn empty_body() {
        let framed = frame("");
        let mut cursor = Cursor::new(framed);
        let result = read_message(&mut cursor).unwrap();
        assert_eq!(result, "");
    }

    #[test]
    fn unicode_body() {
        let body = r#"{"emoji":"🚀","text":"héllo"}"#;
        let mut buf = Vec::new();
        write_message(&mut buf, body).unwrap();
        let mut cursor = Cursor::new(buf);
        let result = read_message(&mut cursor).unwrap();
        assert_eq!(result, body);
    }

    #[test]
    fn case_insensitive_header() {
        let data = b"content-length: 13\r\n\r\n{\"test\":true}";
        let mut cursor = Cursor::new(data.to_vec());
        let result = read_message(&mut cursor).unwrap();
        assert_eq!(result, r#"{"test":true}"#);
    }

    #[test]
    fn message_too_large() {
        let huge = MAX_MESSAGE_BYTES + 1;
        let header = format!("Content-Length: {huge}\r\n\r\n");
        let mut cursor = Cursor::new(header.into_bytes());
        let err = read_message(&mut cursor).unwrap_err();
        assert!(matches!(err, FramingError::MessageTooLarge(_)));
    }

    #[test]
    fn additional_headers_skipped() {
        let data = b"Content-Length: 13\r\nContent-Type: application/json\r\n\r\n{\"test\":true}";
        let mut cursor = Cursor::new(data.to_vec());
        let result = read_message(&mut cursor).unwrap();
        assert_eq!(result, r#"{"test":true}"#);
    }
}
