#!/usr/bin/env ruby
require 'json'
require 'open3'
require 'tmpdir'
require 'fileutils'
require 'digest'
require 'rbconfig'

ROOT = File.expand_path('..', __dir__)
BRIDGE = File.join(ROOT, 'Scripts/abslayer-tool.rb')

def assert(condition, message)
  raise message unless condition
end

def call_json(*command, input:)
  out, err, status = Open3.capture3(*command, stdin_data: JSON.generate(input))
  value = JSON.parse(out)
  assert(status.success? && value['ok'], "Request failed: #{out}\n#{err}")
  value
end

def wait_for(seconds: 8)
  deadline = Process.clock_gettime(Process::CLOCK_MONOTONIC) + seconds
  loop do
    value = yield
    return value if value
    raise 'Timed out waiting for expected state' if Process.clock_gettime(Process::CLOCK_MONOTONIC) > deadline
    sleep 0.05
  end
end

pins = Digest::SHA256.file(File.join(ROOT, 'Package.resolved')).hexdigest
host, errors, status = Open3.capture3(RbConfig.ruby, File.join(ROOT, 'Scripts/build-harness.rb'))
$stderr.write(errors)
assert(status.success?, 'Controller build failed')
host = host.strip

Dir.mktmpdir('abslayer-capabilities-') do |temporary|
  environment = {
    'ABSLAYER_RUBY' => RbConfig.ruby,
    'ABSLAYER_LAGUNA_AUTHORIZED_SCREEN_WORKER' => __FILE__,
    'ABSLAYER_LAGUNA_SERVER' => __FILE__,
    'ABSLAYER_LAGUNA_MODEL' => __FILE__,
    'ABSLAYER_LAGUNA_AUTHORIZED_DATASET' => __FILE__,
    'ABSLAYER_LAGUNA_AUTHORIZED_MANIFEST' => __FILE__,
    'ABSLAYER_LAGUNA_AUTHORIZED_SCREEN_OUTPUT' => File.join(temporary, 'screen'),
    'ABSLAYER_LAGUNA_REVIEWED_VECTOR_WORKER' => __FILE__,
    'ABSLAYER_LAGUNA_REVIEWED_SELECTION' => __FILE__,
    'ABSLAYER_LAGUNA_REVIEWED_SCREEN_RESULTS' => __FILE__,
    'ABSLAYER_LAGUNA_GENERATOR' => __FILE__,
    'ABSLAYER_LAGUNA_REVIEWED_VECTOR_OUTPUT' => File.join(temporary, 'vector'),
    # Supplying the withdrawn configuration must not restore its operations.
    'ABSLAYER_LAGUNA_WORKER' => __FILE__,
    'ABSLAYER_LAGUNA_DATASET' => __FILE__,
    'ABSLAYER_LAGUNA_OUTPUT' => File.join(temporary, 'legacy-vector'),
    'ABSLAYER_LAGUNA_SCREEN_WORKER' => __FILE__,
    'ABSLAYER_LAGUNA_VECTOR' => __FILE__,
    'ABSLAYER_LAGUNA_SCREEN_DATASET' => __FILE__,
    'ABSLAYER_LAGUNA_SCREEN_OUTPUT' => File.join(temporary, 'legacy-screen')
  }
  response = call_json(environment, host, 'request', ROOT, File.join(temporary, 'state'), RbConfig.ruby,
                       input: {method: 'capabilities'})
  operations = response.fetch('operations')
  assert(operations.include?('laguna_authorized_screen'), 'Authorized screen was not registered')
  assert(operations.include?('laguna_reviewed_vector'), 'Reviewed vector was not registered')
  assert(!operations.include?('laguna_control_vector') && !operations.include?('laguna_vector_screen'),
         'Withdrawn Laguna operations remain callable')
  puts 'PASS: replacement Laguna operations register and withdrawn operations stay unavailable'
end

Dir.mktmpdir('abslayer-bridge-') do |temporary|
  environment = {'ABSLAYER_STATE_DIR' => File.join(temporary, 'state')}
  request = {method: 'submit', operation: 'workspace_preflight', idempotencyKey: 'shared-client-run', timeoutSeconds: 60}
  first = call_json(environment, RbConfig.ruby, BRIDGE, input: request).fetch('job')
  # Every call is a new process, just as switching command tools/clients is.
  second = call_json(environment, RbConfig.ruby, BRIDGE, input: request).fetch('job')
  assert(first['id'] == second['id'], 'Repeated bridge call duplicated the job')
  final = wait_for(seconds: 60) do
    job = call_json(environment, RbConfig.ruby, BRIDGE, input: {method: 'status', jobID: first['id']}).fetch('job')
    %w[queued running].include?(job['state']) ? nil : job
  end
  assert(final['state'] == 'completed', "Real SwiftPM preflight did not complete: #{final}")
  assert(final['acceptance'] == 'not_evaluated', 'Workspace success became scientific acceptance')
  result = call_json(environment, RbConfig.ruby, BRIDGE, input: {method: 'evidence', jobID: first['id'], stream: 'stdout', limit: 128})
  assert(result['text'].bytesize <= 128 && result['artifact']['sha256'].length == 64, 'Evidence was not bounded and identified')
  puts 'PASS: real SwiftPM preflight, detached execution, reconnect, deduplication, bounded evidence'
end

Dir.mktmpdir('abslayer-crash-') do |temporary|
  workspace = File.realpath(temporary)
  state = File.join(workspace, 'state')
  File.write(File.join(workspace, 'Package.swift'), 'fixture manifest identity')
  File.write(File.join(workspace, 'Package.resolved'), 'fixture pins identity')
  worker = File.join(workspace, 'fake-swift')
  File.write(worker, "#!/bin/sh\nsleep 2\nprintf '%s' '{\"name\":\"fixture\",\"toolsVersion\":{},\"targets\":[]}'\n")
  File.chmod(0700, worker)
  command = [host, 'request', workspace, state, worker]
  first = call_json(*command, input: {method: 'submit', idempotencyKey: 'crash-one'}).fetch('job')
  supervisor = Process.spawn(host, 'drain', workspace, state, worker, out: File::NULL, err: File::NULL, pgroup: true)
  begin
    wait_for do
      record = JSON.parse(File.read(File.join(state, 'state.json'))).fetch('jobs').find { |job| job['id'] == first['id'] }
      raise "Fixture failed before launch: #{record['failure']}" if %w[failed interrupted cancelled].include?(record['state'])
      record['workerPID']
    end
    Process.kill('KILL', supervisor)
    Process.wait(supervisor)
    supervisor = nil
    second = call_json(*command, input: {method: 'submit', idempotencyKey: 'crash-two'}).fetch('job')
    # An orphan still holding the inherited lease prevents a new supervisor
    # from marking it interrupted or launching another worker concurrently.
    assert(system(host, 'drain', workspace, state, worker), 'Recovery host failed')
    blocked = call_json(*command, input: {method: 'status', jobID: second['id']}).fetch('job')
    assert(blocked['state'] == 'queued', 'New worker overlapped the orphan')
    recovered, final = wait_for(seconds: 10) do
      assert(system(host, 'drain', workspace, state, worker), 'Post-orphan recovery failed')
      first_status = call_json(*command, input: {method: 'status', jobID: first['id']}).fetch('job')
      second_status = call_json(*command, input: {method: 'status', jobID: second['id']}).fetch('job')
      if first_status['state'] == 'interrupted' && second_status['state'] == 'completed'
        [first_status, second_status]
      elsif %w[failed cancelled completed].include?(first_status['state']) ||
            %w[failed cancelled interrupted].include?(second_status['state'])
        raise "Unexpected recovery state: #{first_status} / #{second_status}"
      end
    end
    assert(recovered['state'] == 'interrupted', "Crashed job was falsely completed or automatically retried: #{recovered}")
    assert(final['state'] == 'completed', "Queued job did not resume after the lease cleared: #{final}")
    puts 'PASS: supervisor crash, inherited orphan lease, interrupted-state recovery, queued-work continuation'
  ensure
    if supervisor
      Process.kill('KILL', supervisor) rescue Errno::ESRCH
      Process.wait(supervisor) rescue Errno::ECHILD
    end
  end
end

assert(Digest::SHA256.file(File.join(ROOT, 'Package.resolved')).hexdigest == pins, 'Controller work changed model dependency pins')
puts 'PASS: model dependency pins preserved'
