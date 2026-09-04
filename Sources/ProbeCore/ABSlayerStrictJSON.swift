import Foundation

/// Rejects duplicate object keys (including escape-equivalent spellings) at
/// every nesting level before Foundation decoders apply last-key-wins policy.
enum ABSlayerStrictJSON {
    static func validateNoDuplicateKeys(_ data: Data) throws {
        var parser = Parser(bytes: Array(data))
        try parser.parseDocument()
    }

    private struct Parser {
        let bytes: [UInt8]
        var index = 0

        mutating func parseDocument() throws {
            skipWhitespace()
            try parseValue(depth: 0)
            skipWhitespace()
            guard index == bytes.count else { throw Error.invalidJSON }
        }

        mutating func parseValue(depth: Int) throws {
            guard depth <= 128, index < bytes.count else {
                throw Error.invalidJSON
            }
            switch bytes[index] {
            case 0x7b: try parseObject(depth: depth + 1) // {
            case 0x5b: try parseArray(depth: depth + 1) // [
            case 0x22: _ = try parseString()
            case 0x74: try consumeLiteral("true")
            case 0x66: try consumeLiteral("false")
            case 0x6e: try consumeLiteral("null")
            default: try parseNumber()
            }
        }

        mutating func parseObject(depth: Int) throws {
            try expect(0x7b)
            skipWhitespace()
            if consume(0x7d) { return }
            var keys = Set<String>()
            while true {
                let key = try parseString()
                guard keys.insert(key).inserted else {
                    throw Error.duplicateKey(key)
                }
                skipWhitespace()
                try expect(0x3a)
                skipWhitespace()
                try parseValue(depth: depth)
                skipWhitespace()
                if consume(0x7d) { return }
                try expect(0x2c)
                skipWhitespace()
            }
        }

        mutating func parseArray(depth: Int) throws {
            try expect(0x5b)
            skipWhitespace()
            if consume(0x5d) { return }
            while true {
                try parseValue(depth: depth)
                skipWhitespace()
                if consume(0x5d) { return }
                try expect(0x2c)
                skipWhitespace()
            }
        }

        mutating func parseString() throws -> String {
            guard index < bytes.count, bytes[index] == 0x22 else {
                throw Error.invalidJSON
            }
            let start = index
            index += 1
            while index < bytes.count {
                let byte = bytes[index]
                if byte == 0x22 {
                    index += 1
                    let token = Data(bytes[start ..< index])
                    guard let value = try? JSONDecoder().decode(String.self, from: token)
                    else { throw Error.invalidJSON }
                    return value
                }
                guard byte >= 0x20 else { throw Error.invalidJSON }
                if byte == 0x5c {
                    index += 1
                    guard index < bytes.count else { throw Error.invalidJSON }
                    if bytes[index] == 0x75 {
                        guard index + 4 < bytes.count else { throw Error.invalidJSON }
                        for scalar in bytes[(index + 1) ... (index + 4)] {
                            guard Self.isHex(scalar) else { throw Error.invalidJSON }
                        }
                        index += 4
                    } else {
                        guard [0x22, 0x5c, 0x2f, 0x62, 0x66, 0x6e, 0x72, 0x74]
                            .contains(bytes[index])
                        else { throw Error.invalidJSON }
                    }
                }
                index += 1
            }
            throw Error.invalidJSON
        }

        mutating func parseNumber() throws {
            let start = index
            while index < bytes.count,
                  ![0x20, 0x09, 0x0a, 0x0d, 0x2c, 0x5d, 0x7d].contains(bytes[index])
            { index += 1 }
            guard index > start else { throw Error.invalidJSON }
            let token = Data(bytes[start ..< index])
            guard (try? JSONSerialization.jsonObject(
                with: token, options: [.fragmentsAllowed])) is NSNumber
            else { throw Error.invalidJSON }
        }

        mutating func consumeLiteral(_ value: StaticString) throws {
            let literal = Array(String(describing: value).utf8)
            guard index + literal.count <= bytes.count,
                  Array(bytes[index ..< index + literal.count]) == literal
            else { throw Error.invalidJSON }
            index += literal.count
        }

        mutating func skipWhitespace() {
            while index < bytes.count,
                  [0x20, 0x09, 0x0a, 0x0d].contains(bytes[index])
            { index += 1 }
        }

        mutating func expect(_ byte: UInt8) throws {
            guard consume(byte) else { throw Error.invalidJSON }
        }

        mutating func consume(_ byte: UInt8) -> Bool {
            guard index < bytes.count, bytes[index] == byte else { return false }
            index += 1
            return true
        }

        static func isHex(_ byte: UInt8) -> Bool {
            (0x30 ... 0x39).contains(byte)
                || (0x41 ... 0x46).contains(byte)
                || (0x61 ... 0x66).contains(byte)
        }
    }

    enum Error: Swift.Error, Equatable {
        case invalidJSON
        case duplicateKey(String)
    }
}
