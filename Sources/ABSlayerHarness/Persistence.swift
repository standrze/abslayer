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
    static func binding(_ url: URL, maxBytes: Int? = nil,
                        _didReadChunk: ((Int) throws -> Void)? = nil) throws -> FileBinding {
        try inspect(url, maxBytes: maxBytes, captureData: false,
                    _didReadChunk: _didReadChunk).binding
    }
    static func verifiedData(_ url: URL, matching expected: FileBinding, maxBytes: Int,
                             _didReadChunk: ((Int) throws -> Void)? = nil) throws -> Data {
        let inspected = try inspect(url, maxBytes: maxBytes, captureData: true,
                                    _didReadChunk: _didReadChunk)
        guard inspected.binding == expected else {
            throw HarnessError("artifact_changed", "Recorded output bytes no longer match their identity.")
        }
        return inspected.data
    }
    private static func inspect(_ url: URL, maxBytes: Int?, captureData: Bool,
                                _didReadChunk: ((Int) throws -> Void)?) throws -> (binding: FileBinding, data: Data) {
        let fd = open(url.path, O_RDONLY | O_CLOEXEC | O_NOFOLLOW)
        guard fd >= 0 else {
            throw HarnessError("binding_failed", "Cannot open a bound regular file: \(url.path)")
        }
        defer { close(fd) }
        var information = stat()
        guard fstat(fd, &information) == 0,
              (information.st_mode & mode_t(S_IFMT)) == mode_t(S_IFREG) else {
            throw HarnessError("binding_failed", "Bound path is not a regular file: \(url.path)")
        }
        if let maxBytes {
            guard maxBytes >= 0 else {
                throw HarnessError("binding_failed", "Bound file has an invalid byte limit: \(url.path)")
            }
            guard information.st_size >= 0, information.st_size <= off_t(maxBytes) else {
                throw HarnessError("binding_failed", "Bound file exceeds its byte limit: \(url.path)")
            }
        }
        var hash = SHA256(); var count = 0; var captured = Data()
        var buffer = [UInt8](repeating: 0, count: 1_048_576)
        while true {
            let amount = buffer.withUnsafeMutableBytes { bytes in
                systemRead(fd, bytes.baseAddress!, bytes.count)
            }
            if amount < 0 && errno == EINTR { continue }
            guard amount >= 0 else {
                throw HarnessError("binding_failed", "Cannot read a bound regular file: \(url.path)")
            }
            if amount == 0 { break }
            guard amount <= Int.max - count,
                  maxBytes.map({ amount <= $0 - count }) ?? true else {
                throw HarnessError("binding_failed", "Bound file exceeds its byte limit: \(url.path)")
            }
            let chunk = Data(buffer[0..<amount])
            hash.update(data: chunk)
            if captureData { captured.append(chunk) }
            count += amount
            try _didReadChunk?(count)
        }
        var finalInformation = stat()
        guard fstat(fd, &finalInformation) == 0,
              stableSnapshot(information, finalInformation),
              information.st_size == off_t(count) else {
            throw HarnessError("binding_failed", "Bound file changed while its identity was recorded: \(url.path)")
        }
        let binding = FileBinding(path: url.path,
                                  sha256: hash.finalize().map { String(format: "%02x", $0) }.joined(),
                                  bytes: count)
        return (binding, captured)
    }

    private static func stableSnapshot(_ before: stat, _ after: stat) -> Bool {
        let common = before.st_dev == after.st_dev && before.st_ino == after.st_ino &&
            before.st_mode == after.st_mode && before.st_nlink == after.st_nlink &&
            before.st_uid == after.st_uid && before.st_gid == after.st_gid &&
            before.st_size == after.st_size
        #if canImport(Darwin)
        return common &&
            before.st_mtimespec.tv_sec == after.st_mtimespec.tv_sec &&
            before.st_mtimespec.tv_nsec == after.st_mtimespec.tv_nsec &&
            before.st_ctimespec.tv_sec == after.st_ctimespec.tv_sec &&
            before.st_ctimespec.tv_nsec == after.st_ctimespec.tv_nsec
        #else
        return common &&
            before.st_mtim.tv_sec == after.st_mtim.tv_sec &&
            before.st_mtim.tv_nsec == after.st_mtim.tv_nsec &&
            before.st_ctim.tv_sec == after.st_ctim.tv_sec &&
            before.st_ctim.tv_nsec == after.st_ctim.tv_nsec
        #endif
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

private func systemRead(_ fd: Int32, _ bytes: UnsafeMutableRawPointer, _ count: Int) -> Int {
    #if canImport(Darwin)
    Darwin.read(fd, bytes, count)
    #else
    Glibc.read(fd, bytes, count)
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
