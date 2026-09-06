#!/usr/bin/env ruby
# Shared Pool/Codex command bridge. All job policy/state lives in Swift.
require 'open3'
require 'json'
require 'rbconfig'
require 'fileutils'

begin
  raise 'Pass one JSON request as an argument, or send it on stdin.' if ARGV.length > 1
  input = ARGV.empty? ? STDIN.read(16_385) : ARGV.first
  raise 'Request exceeds 16 KiB.' if input.bytesize > 16_384
  root = File.realpath(File.expand_path('..', __dir__))
  build, errors, status = Open3.capture3(RbConfig.ruby, File.join(__dir__, 'build-harness.rb'))
  $stderr.write(errors)
  raise 'The Swift controller could not be built.' unless status.success?
  host = build.strip
  swift = ENV.fetch('ABSLAYER_SWIFT', 'swift')
  unless swift.include?(File::SEPARATOR)
    swift = ENV.fetch('PATH', '').split(File::PATH_SEPARATOR).map { |dir| File.join(dir, swift) }.find { |path| File.executable?(path) && File.file?(path) }
  end
  raise 'Swift executable not found.' unless swift && File.executable?(swift)
  swift = File.expand_path(swift)
  state = File.expand_path(ENV.fetch('ABSLAYER_STATE_DIR', File.join(root, '.abslayer', 'state')))
  controller_environment = {'ABSLAYER_RUBY' => RbConfig.ruby}
  output, errors, status = Open3.capture3(controller_environment, host, 'request', root, state, swift,
                                          stdin_data: input, chdir: root)
  $stderr.write(errors)
  response = JSON.parse(output)
  if status.success? && response['ok'] && response['needsDrain']
    log = File.open(File.join(state, 'host.log'), File::WRONLY | File::APPEND | File::CREAT, 0600)
    begin
      pid = Process.spawn(controller_environment, host, 'drain', root, state, swift,
                          in: File::NULL, out: log, err: log, pgroup: true, chdir: root)
      Process.detach(pid)
    ensure
      log.close
    end
  end
  STDOUT.write(output)
  exit(status.exitstatus || 2)
rescue StandardError => error
  puts JSON.generate(schemaVersion: 1, ok: false, code: 'bridge_error', message: error.message)
  exit 2
end
