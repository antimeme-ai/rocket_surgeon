pub fn encode_varint(mut value: u64, buf: &mut Vec<u8>) {
    loop {
        let byte = (value & 0x7F) as u8;
        value >>= 7;
        if value == 0 {
            buf.push(byte);
            break;
        }
        buf.push(byte | 0x80);
    }
}

pub fn field1_tag_and_length(payload_len: usize, buf: &mut Vec<u8>) {
    buf.push(0x0A);
    encode_varint(payload_len as u64, buf);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_varint_zero() {
        let mut buf = Vec::new();
        encode_varint(0, &mut buf);
        assert_eq!(buf, [0x00]);
    }

    #[test]
    fn encode_varint_one_byte() {
        let mut buf = Vec::new();
        encode_varint(0x7F, &mut buf);
        assert_eq!(buf, [0x7F]);
    }

    #[test]
    fn encode_varint_two_bytes() {
        let mut buf = Vec::new();
        encode_varint(300, &mut buf);
        assert_eq!(buf, [0xAC, 0x02]);
    }

    #[test]
    fn encode_varint_max_u64() {
        let mut buf = Vec::new();
        encode_varint(u64::MAX, &mut buf);
        assert_eq!(buf.len(), 10);
        assert_eq!(
            buf,
            [0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x01]
        );
    }

    #[test]
    fn field1_tag_and_length_small() {
        let mut buf = Vec::new();
        field1_tag_and_length(5, &mut buf);
        assert_eq!(buf, [0x0A, 0x05]);
    }

    #[test]
    fn field1_tag_and_length_large() {
        let mut buf = Vec::new();
        field1_tag_and_length(300, &mut buf);
        assert_eq!(buf, [0x0A, 0xAC, 0x02]);
    }
}
