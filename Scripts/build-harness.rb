#!/usr/bin/env ruby
# Isolated SwiftPM build of the real sources. Never resolves or rewrites the
# product's model dependency pins. Ruby owns build plumbing only.
require 'fileutils'
require 'digest'
require 'open3'

begin
  mode = ARGV.fetch(0, 'build')
  raise 'Usage: ruby Scripts/build-harness.rb [build|test]' unless %w[build test].include?(mode) && ARGV.length <= 1
  root = File.expand_path('..', __dir__)
  build_root = File.join(root, '.build-harness')
  package = File.join(build_root, 'package')
  FileUtils.mkdir_p(package)
  swift = ENV.fetch('ABSLAYER_SWIFT', 'swift')
  source_paths = [File.join(root, 'Package.swift'), File.expand_path(__FILE__)]
  source_paths += Dir.glob(File.join(root, '{Sources/ABSlayerHarness,Sources/ABSlayerJobHost,Tests/ABSlayerHarnessTests}/**/*.swift')).sort
  digest = Digest::SHA256.new
  source_paths.each { |path| digest.update(path); digest.update(File.binread(path)) }
  version, err, status = Open3.capture3(swift, '--version')
  raise err unless status.success?
  digest.update(version)
  fingerprint = digest.hexdigest
  File.open(File.join(build_root, 'build.lock'), File::RDWR | File::CREAT, 0600) do |lock|
    lock.flock(File::LOCK_EX)
    marker = File.join(build_root, 'fingerprint')
    binary_marker = File.join(build_root, 'binary-path')
    binary = File.file?(binary_marker) ? File.read(binary_marker).strip : ''
    if mode == 'build' && File.file?(marker) && File.read(marker) == fingerprint && File.executable?(binary)
      puts binary
      exit 0
    end
    FileUtils.cp(File.join(root, 'Package.swift'), File.join(package, 'Package.swift'))
    %w[Sources Tests].each do |name|
      destination = File.join(package, name)
      source = File.join(root, name)
      if File.symlink?(destination)
        raise "Unexpected build link: #{destination}" unless File.readlink(destination) == source
      elsif File.exist?(destination)
        raise "Unexpected build path: #{destination}"
      else
        File.symlink(source, destination)
      end
    end
    settings = {'ABSLAYER_HARNESS_ONLY' => '1',
                'CLANG_MODULE_CACHE_PATH' => File.join(build_root, 'module-cache'),
                'SWIFT_MODULECACHE_PATH' => File.join(build_root, 'module-cache')}
    flags = ['--disable-sandbox', '--package-path', package,
             '--cache-path', File.join(build_root, 'cache'),
             '--scratch-path', File.join(build_root, 'build')]
    command = mode == 'test' ? ['test'] : ['build']
    command += flags
    command += ['--product', 'abslayer-job-host'] if mode == 'build'
    unless system(settings, swift, *command, out: $stderr, err: $stderr)
      raise "Swift harness #{mode} failed"
    end
    output, err, status = Open3.capture3(settings, swift, 'build', *flags, '--show-bin-path')
    raise err unless status.success?
    binary = File.join(output.strip, 'abslayer-job-host')
    File.write(marker, fingerprint)
    File.write(binary_marker, binary)
    puts binary
  end
rescue StandardError => error
  warn error.message
  exit 1
end
