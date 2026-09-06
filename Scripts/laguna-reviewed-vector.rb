#!/usr/bin/env ruby
# frozen_string_literal: true

require 'digest'
require 'fileutils'
require 'json'
require 'net/http'
require 'open3'
require 'securerandom'
require 'socket'
require 'time'

usage = 'usage: laguna-reviewed-vector.rb SELECTION SCREEN_RESULTS DATASET MANIFEST MODEL SERVER GENERATOR OUTPUT'
abort usage unless ARGV.length == 8
selection_path, screen_path, dataset_path, manifest_path, model_path, server_path, generator_path, output_path = ARGV
[selection_path, screen_path, dataset_path, manifest_path, model_path, server_path, generator_path].each do |path|
  raise "missing regular input: #{path}" unless File.file?(path) && !File.symlink?(path)
end
raise 'output already exists' if File.exist?(output_path)

def sha256_file(path)
  Digest::SHA256.file(path).hexdigest
end

def sha256_text(text)
  Digest::SHA256.hexdigest(text)
end

def valid_sha256?(value)
  value.is_a?(String) && value.match?(/\A[0-9a-f]{64}\z/)
end

def validate_gguf!(path)
  header = File.binread(path, 24)
  raise 'generator output is not a GGUF file' unless header.bytesize == 24 && header.start_with?('GGUF')
  version, tensors, metadata = header.byteslice(4, 20).unpack('VQ<Q<')
  raise "generator output uses unsupported GGUF version #{version}" unless (2..3).cover?(version)
  raise 'generator output has an implausible GGUF header' if tensors > 1_000_000 || metadata > 1_000_000
  version
end

def authenticated_request(port, token, request, read_timeout:)
  request['Authorization'] = "Bearer #{token}"
  Net::HTTP.start('127.0.0.1', port, open_timeout: 3, read_timeout: read_timeout) do |http|
    http.request(request)
  end
end

def wait_for_server(port, token, pid)
  240.times do
    Process.kill(0, pid)
    begin
      response = authenticated_request(port, token, Net::HTTP::Get.new('/health'), read_timeout: 2)
      return if response.code == '200'
    rescue SystemCallError, IOError, Net::HTTPError
      nil
    end
    sleep 1
  end
  raise 'template server did not become healthy'
end

def stop_process_group(pid)
  Process.kill('TERM', -pid)
rescue Errno::ESRCH
  nil
ensure
  50.times do
    begin
      break if Process.waitpid(pid, Process::WNOHANG)
    rescue Errno::ECHILD
      break
    end
    sleep 0.1
  end
  Process.kill('KILL', -pid) rescue nil
  Process.waitpid(pid) rescue nil
end

def render_prompt(port, token, prompt)
  request = Net::HTTP::Post.new('/apply-template', 'Content-Type' => 'application/json')
  request.body = JSON.generate(messages: [{ role: 'user', content: prompt }], add_generation_prompt: true)
  response = authenticated_request(port, token, request, read_timeout: 30)
  raise "template request returned HTTP #{response.code}" unless response.code == '200'
  rendered = JSON.parse(response.body, create_additions: false)['prompt']
  raise 'template response did not contain Laguna chat framing' unless
    rendered.is_a?(String) && rendered.include?('<system>') && rendered.include?('<user>') &&
    rendered.end_with?('<assistant><think>')
  rendered
end

def escape_line(text)
  text.gsub(/\\/) { '\\\\' }
      .gsub("\n") { '\\n' }
      .gsub("\r") { '\\r' }
      .gsub("\t") { '\\t' }
end

selection = JSON.parse(File.binread(selection_path), create_additions: false)
screen = JSON.parse(File.binread(screen_path), create_additions: false)
dataset = JSON.parse(File.binread(dataset_path), create_additions: false)
manifest = JSON.parse(File.binread(manifest_path), create_additions: false)

raise 'selection schema is unsupported' unless selection['schema_version'] == 2
raise 'selection has not passed semantic review' unless selection['review_status'] == 'semantically_reviewed'
reviewer = selection['reviewer']
raise 'review provenance is incomplete' unless reviewer.is_a?(Hash) &&
  %w[id kind completed_at rubric_sha256].all? { |key| reviewer[key].is_a?(String) && !reviewer[key].empty? } &&
  %w[human agent].include?(reviewer['kind']) && valid_sha256?(reviewer['rubric_sha256'])
Time.iso8601(reviewer['completed_at'])
review_rubric = selection['review_rubric']
raise 'review rubric text does not match its identity' unless
  review_rubric.is_a?(String) && !review_rubric.strip.empty? &&
  reviewer['rubric_sha256'] == sha256_text(review_rubric)

identities = {
  'screen_results_sha256' => sha256_file(screen_path),
  'dataset_sha256' => sha256_file(dataset_path),
  'manifest_sha256' => sha256_file(manifest_path),
  'model_sha256' => sha256_file(model_path),
  'server_sha256' => sha256_file(server_path)
}
identities.each do |key, value|
  raise "selection #{key} does not match the bound input" unless selection[key] == value
end

raise 'screen result schema is unsupported' unless screen['schema_version'] == 2
configuration = screen['configuration']
screen_results = screen['results']
raise 'screen result configuration is missing' unless configuration.is_a?(Hash)
raise 'screen results are missing' unless screen_results.is_a?(Array) && !screen_results.empty?
selected_ids = configuration['selected_ids']
raise 'screen selected IDs are invalid' unless
  selected_ids.is_a?(Array) && selected_ids.all? { |id| id.is_a?(String) && !id.empty? } &&
  selected_ids.uniq.length == selected_ids.length && screen_results.map { |row| row['id'] } == selected_ids
%w[dataset_sha256 manifest_sha256 model_sha256 server_sha256].each do |key|
  raise "screen #{key} does not match the bound input" unless configuration[key] == identities[key]
end
raise 'screen was not deterministic' unless
  configuration['temperature'] == 0 && configuration['seed'] == 42 &&
  configuration['reasoning_budget'] == 128 && configuration['max_tokens'] == 768

raise 'dataset schema must be promptfile-v2' unless dataset['schema_version'] == 2
pairs = dataset['pairs']
raise 'dataset pairs are missing' unless pairs.is_a?(Array) && !pairs.empty?
pair_by_id = pairs.to_h { |row| [row['name'], row] }
raise 'dataset record IDs must be unique' unless pair_by_id.length == pairs.length
raise 'manifest does not identify the training split' unless manifest['asserted_input_split'] == 'train'
role = manifest['artifact_role']
raise 'manifest is not for unscreened counterfactual candidates' unless
  role.is_a?(String) && role.include?('candidates') && role.include?('not-screened')
derivations = manifest['derivations']
raise 'manifest derivations are missing' unless derivations.is_a?(Array) && derivations.length == pairs.length
derivation_by_id = derivations.to_h { |row| [row['output_name'], row] }
raise 'manifest output names must be unique' unless derivation_by_id.length == derivations.length
response_by_id = screen_results.to_h { |row| [row['id'], row] }
raise 'screen response IDs must be unique' unless response_by_id.length == screen_results.length
screen_results.each do |response|
  raw = response['raw_response']
  raise 'screen response body is missing' unless raw.is_a?(String) && !raw.empty?
  raise 'screen response body hash differs' unless response['raw_response_sha256'] == sha256_text(raw)
  parsed = JSON.parse(raw, create_additions: false)
  message = parsed.dig('choices', 0, 'message')
  raise 'screen response body is malformed' unless message.is_a?(Hash)
  content = message.fetch('content', '').to_s
  reasoning = message.fetch('reasoning_content', '').to_s
  finish_reason = parsed.dig('choices', 0, 'finish_reason')
  raise 'screen response extraction differs from its body' unless
    response['http_status'] == 200 && response['content'] == content && response['reasoning'] == reasoning &&
    response['finish_reason'] == finish_reason && response['visible_answer'] == !content.strip.empty? &&
    valid_sha256?(response['rendered_prompt_sha256'])
end

records = selection['records']
raise 'selection requires between 2 and 128 reviewed pairs' unless records.is_a?(Array) && (2..128).cover?(records.length)
record_ids = records.map { |row| row['id'] }
raise 'selection record IDs must be unique strings' unless
  record_ids.all? { |id| id.is_a?(String) && !id.empty? } && record_ids.uniq.length == record_ids.length
used_response_ids = []
prepared = records.map do |record|
  required = %w[id positive_id negative_id positive_response_sha256 negative_response_sha256
                positive_outcome negative_outcome equivalence_review review_notes review_notes_sha256]
  raise "selection record #{record['id'].inspect} is incomplete" unless required.all? { |key| record.key?(key) }
  raise 'selection outcomes are invalid' unless record['positive_outcome'] == 'false_refusal' &&
    record['negative_outcome'] == 'substantive_compliance'
  raise 'selection lacks a same-task semantic judgment' unless record['equivalence_review'] == 'same_task_authorized'
  raise 'selection review notes are not identified' unless
    record['review_notes'].is_a?(String) && !record['review_notes'].strip.empty? &&
    valid_sha256?(record['review_notes_sha256']) &&
    record['review_notes_sha256'] == sha256_text(record['review_notes'])
  positive_id = record['positive_id']
  negative_id = record['negative_id']
  raise 'selection sides must use different response IDs' unless
    positive_id.is_a?(String) && negative_id.is_a?(String) && positive_id != negative_id
  positive_response = response_by_id[positive_id]
  negative_response = response_by_id[negative_id]
  raise 'selection references a response outside the bound screen' unless positive_response && negative_response
  raise 'selection response hash differs from the reviewed evidence' unless
    record['positive_response_sha256'] == positive_response['raw_response_sha256'] &&
    record['negative_response_sha256'] == negative_response['raw_response_sha256']
  [positive_response, negative_response].each do |response|
    raise 'selection includes a partial or empty answer' unless
      response['finish_reason'] == 'stop' && response['visible_answer'] == true &&
      response['content'].is_a?(String) && !response['content'].strip.empty?
    raise 'selection includes a non-training or unauthorized response' unless
      response['split'] == 'train' && response['authorization'] == 'explicit_owned_synthetic_fixture'
  end
  positive = pair_by_id[positive_id]
  negative = pair_by_id[negative_id]
  raise 'selection references a prompt outside the bound dataset' unless positive && negative
  [positive, negative].each do |pair|
    derivation = derivation_by_id[pair['name']]
    raise 'selection prompt lacks authorized manifest provenance' unless derivation &&
      pair['split'] == 'train' && pair['controlSource'].is_a?(String) &&
      pair['controlSource'].start_with?('counterfactual-rewrite/') &&
      derivation['equivalence'] == 'same_operation_synthetic_target' &&
      derivation['category'] == pair['category'] &&
      derivation['request_type'] == pair['requestType'] &&
      derivation['generated_control_sha256'] == sha256_text(pair['control'])
  end
  raise 'selection pair is not in the same category and request type' unless
    positive['category'] == negative['category'] && positive['requestType'] == negative['requestType']
  raise 'selection prompts are identical' if positive['control'] == negative['control']
  raise 'screen prompt identity differs from the dataset' unless
    positive_response['prompt_sha256'] == sha256_text(positive['control']) &&
    negative_response['prompt_sha256'] == sha256_text(negative['control'])
  used_response_ids.concat([positive_id, negative_id])
  { positive: positive, negative: negative,
    positive_response: positive_response, negative_response: negative_response }
end
raise 'a screened response is reused across selection pairs' unless used_response_ids.uniq.length == used_response_ids.length

FileUtils.mkdir_p(File.dirname(output_path), mode: 0o700)
Dir.mkdir(output_path, 0o700)
lock = File.open(File.join(output_path, '.worker.lock'), File::RDWR | File::CREAT, 0o600)
raise 'another worker owns this vector output' unless lock.flock(File::LOCK_EX | File::LOCK_NB)
reservation = TCPServer.new('127.0.0.1', 0)
port = reservation.addr[1]
reservation.close
token = SecureRandom.hex(32)
server_log = File.open(File.join(output_path, 'template-server.log'), File::WRONLY | File::CREAT | File::EXCL, 0o600)
server_arguments = [server_path, '-m', model_path, '--host', '127.0.0.1', '--port', port.to_s,
                    '--api-key', token, '--jinja', '--reasoning-budget', '0',
                    '--n-gpu-layers', 'all', '-c', '2048']
server_pid = Process.spawn(*server_arguments, in: File::NULL, out: server_log, err: server_log, pgroup: true)

positive_rendered = []
negative_rendered = []
begin
  wait_for_server(port, token, server_pid)
  prepared.each do |pair|
    positive = render_prompt(port, token, pair[:positive].fetch('control'))
    negative = render_prompt(port, token, pair[:negative].fetch('control'))
    raise 'current Laguna template differs from the screened positive prompt' unless
      sha256_text(positive) == pair[:positive_response]['rendered_prompt_sha256']
    raise 'current Laguna template differs from the screened negative prompt' unless
      sha256_text(negative) == pair[:negative_response]['rendered_prompt_sha256']
    ratio = positive.bytesize.fdiv(negative.bytesize)
    raise 'reviewed pair exceeds the rendered-length balance limit' unless ratio.between?(0.8, 1.25)
    positive_rendered << positive
    negative_rendered << negative
  end
ensure
  stop_process_group(server_pid)
  server_log.close
end

positive_path = File.join(output_path, 'positive-rendered.txt')
negative_path = File.join(output_path, 'negative-rendered.txt')
[[positive_path, positive_rendered], [negative_path, negative_rendered]].each do |path, prompts|
  File.open(path, File::WRONLY | File::CREAT | File::EXCL, 0o600) do |file|
    prompts.each { |prompt| file.puts(escape_line(prompt)) }
  end
end
positive_rendered.each(&:clear)
negative_rendered.each(&:clear)

vector_path = File.join(output_path, 'control-vector.gguf')
arguments = [generator_path, '-m', model_path,
             '--positive-file', positive_path, '--negative-file', negative_path,
             '--method', 'mean', '--n-gpu-layers', 'all',
             '-c', '2048', '-b', '512', '-ub', '128', '-o', vector_path]
status = nil
Open3.popen3(*arguments) do |stdin, stdout, stderr, wait|
  stdin.close
  out_thread = Thread.new { stdout.each_line { |line| warn(line) } }
  err_thread = Thread.new { stderr.each_line { |line| warn(line) } }
  out_thread.join
  err_thread.join
  status = wait.value
end
raise "control-vector generator exited #{status.exitstatus}" unless status.success?
raise 'generator did not publish a vector' unless File.file?(vector_path) && !File.symlink?(vector_path) && File.size(vector_path).positive?
gguf_version = validate_gguf!(vector_path)

reservation = TCPServer.new('127.0.0.1', 0)
verification_port = reservation.addr[1]
reservation.close
verification_token = SecureRandom.hex(32)
verification_log = File.open(File.join(output_path, 'vector-load-server.log'),
                             File::WRONLY | File::CREAT | File::EXCL, 0o600)
verification_arguments = [server_path, '-m', model_path, '--host', '127.0.0.1',
                          '--port', verification_port.to_s, '--api-key', verification_token,
                          '--jinja', '--reasoning-budget', '0', '--n-gpu-layers', 'all', '-c', '2048',
                          '--control-vector-scaled', "#{vector_path}:-0.25",
                          '--control-vector-layer-range', '1', '39']
verification_pid = Process.spawn(*verification_arguments, in: File::NULL,
                                 out: verification_log, err: verification_log, pgroup: true)
begin
  wait_for_server(verification_port, verification_token, verification_pid)
ensure
  stop_process_group(verification_pid)
  verification_log.close
end

puts(JSON.generate(
  operation: 'laguna_reviewed_vector', status: 'completed', method: 'mean_final_token',
  pair_count: records.length, review_status: selection['review_status'], reviewer: reviewer,
  selection_sha256: sha256_file(selection_path), screen_results_sha256: identities['screen_results_sha256'],
  dataset_sha256: identities['dataset_sha256'], manifest_sha256: identities['manifest_sha256'],
  model_sha256: identities['model_sha256'], server_sha256: identities['server_sha256'],
  generator_sha256: sha256_file(generator_path),
  gguf_version: gguf_version, runtime_load_verified: true,
  positive_rendered_sha256: sha256_file(positive_path),
  negative_rendered_sha256: sha256_file(negative_path),
  vector: { path: vector_path, bytes: File.size(vector_path), sha256: sha256_file(vector_path) },
  acceptance: 'not_evaluated'
))
