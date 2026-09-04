import Foundation
#if canImport(CryptoKit)
import CryptoKit
#else
import Crypto
#endif
#if canImport(Darwin)
import Darwin
#else
import Glibc
#endif

enum Persistence {
    static func encode<T: Encodable>(_ value: T) throws -> Data {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        return try encoder.encode(value)
    }
    static func digest(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }
    static func binding(_ url: URL) throws -> FileBinding {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        var hash = SHA256(); var count = 0
        while let data = try handle.read(upToCount: 1_048_576), !data.isEmpty {
            hash.update(data: data); count += data.count
        }
        return FileBinding(path: url.path, sha256: hash.finalize().map { String(format: "%02x", $0) }.joined(), bytes: count)
    }
    static func directory(_ url: URL) throws {
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true,
                                               attributes: [.posixPermissions: 0o700])
        let attrs = try FileManager.default.attributesOfItem(atPath: url.path)
        guard attrs[.type] as? FileAttributeType == .typeDirectory else {
            throw HarnessError("invalid_store", "State directory must not be a symlink or a file.")
        }
    }
    static func write<T: Encodable>(_ value: T, to url: URL) throws {
        let data = try encode(value)
        let temporary = url.deletingLastPathComponent().appendingPathComponent(".\(UUID().uuidString).tmp")
        let fd = open(temporary.path, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0o600)
        guard fd >= 0 else { throw HarnessError("store_write", "Cannot create state transaction.") }
        defer { close(fd); unlink(temporary.path) }
        try data.withUnsafeBytes { bytes in
            var offset = 0
            while offset < bytes.count {
                let n = systemWrite(fd, bytes.baseAddress!.advanced(by: offset), bytes.count - offset)
                if n < 0 && errno == EINTR { continue }
                guard n > 0 else { throw HarnessError("store_write", "Cannot write state transaction.") }
                offset += n
            }
        }
        guard fsync(fd) == 0, rename(temporary.path, url.path) == 0 else {
            throw HarnessError("store_write", "Cannot publish state transaction.")
        }
        let parent = open(url.deletingLastPathComponent().path, O_RDONLY)
        if parent >= 0 { _ = fsync(parent); close(parent) }
    }
}

private func systemWrite(_ fd: Int32, _ bytes: UnsafeRawPointer, _ count: Int) -> Int {
    #if canImport(Darwin)
    Darwin.write(fd, bytes, count)
    #else
    Glibc.write(fd, bytes, count)
    #endif
}

final class FileLock {
    private(set) var descriptor: Int32
    init(_ url: URL, nonblocking: Bool = false) throws {
        descriptor = open(url.path, O_RDWR | O_CREAT | O_CLOEXEC | O_NOFOLLOW, 0o600)
        guard descriptor >= 0 else { throw HarnessError("lock_failed", "Cannot open controller lock.") }
        if flock(descriptor, LOCK_EX | (nonblocking ? LOCK_NB : 0)) != 0 {
            close(descriptor); descriptor = -1
            throw HarnessError("busy", "A worker still owns this workspace's execution lease.")
        }
    }
    func release() { if descriptor >= 0 { close(descriptor); descriptor = -1 } }
    deinit { release() }
}
