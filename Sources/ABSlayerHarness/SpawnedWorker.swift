import Foundation
#if canImport(Darwin)
import Darwin
#else
import Glibc
#endif

final class SpawnedWorker {
    let pid: pid_t
    private var finished = false

    init(job: Job, directory: URL, workspace: URL, lease: Int32) throws {
        // posix_spawn file actions, unlike Foundation.Process's descriptor
        // cleanup, explicitly preserve the execution lease in the child.
        #if canImport(Darwin)
        var actions: posix_spawn_file_actions_t?
        var attributes: posix_spawnattr_t?
        #else
        var actions = posix_spawn_file_actions_t()
        var attributes = posix_spawnattr_t()
        #endif
        posix_spawn_file_actions_init(&actions)
        posix_spawnattr_init(&attributes)
        defer { posix_spawn_file_actions_destroy(&actions); posix_spawnattr_destroy(&attributes) }
        posix_spawn_file_actions_addopen(&actions, 0, "/dev/null", O_RDONLY, 0)
        posix_spawn_file_actions_addopen(&actions, 1, directory.appendingPathComponent("stdout.log").path,
                                        O_WRONLY | O_CREAT | O_EXCL, 0o600)
        posix_spawn_file_actions_addopen(&actions, 2, directory.appendingPathComponent("stderr.log").path,
                                        O_WRONLY | O_CREAT | O_EXCL, 0o600)
        posix_spawn_file_actions_adddup2(&actions, lease, 198)
        posix_spawnattr_setflags(&attributes, Int16(POSIX_SPAWN_SETPGROUP))
        posix_spawnattr_setpgroup(&attributes, 0)
        // All production commands carry explicit --package-path; process cwd
        // is inherited from the host, which the bridge starts in the workspace.
        let argumentZero = job.environment["_ABSLAYER_EXECUTABLE_ARGV0"] ?? job.executable.path
        let argv = ([argumentZero] + job.arguments).map { strdup($0) }
        let envp = job.environment.filter { $0.key != "_ABSLAYER_EXECUTABLE_ARGV0" }
            .sorted { $0.key < $1.key }.map { strdup("\($0.key)=\($0.value)") }
        defer { argv.forEach { free($0) }; envp.forEach { free($0) } }
        var args = argv + [nil]; var env = envp + [nil]; var child: pid_t = 0
        let status = posix_spawn(&child, job.executable.path, &actions, &attributes, &args, &env)
        guard status == 0 else { throw HarnessError("spawn_failed", "Worker launch failed: \(String(cString: strerror(status)))") }
        pid = child
    }

    func poll() throws -> Int32? {
        var status: Int32 = 0
        let result = waitpid(pid, &status, WNOHANG)
        if result == 0 { return nil }
        if result < 0 {
            if errno == EINTR { return nil }
            throw HarnessError("wait_failed", "Cannot read worker exit state.")
        }
        finished = true
        let signal = status & 0x7f
        return signal == 0 ? (status >> 8) & 0xff : 128 + signal
    }
    func signal(_ number: Int32) { _ = kill(-pid, number) }
    deinit {
        if !finished {
            signal(SIGKILL)
            var status: Int32 = 0
            while waitpid(pid, &status, 0) < 0 && errno == EINTR {}
        }
    }
}
