#!/usr/bin/env ruby
# frozen_string_literal: true

require 'digest'
require 'fileutils'
require 'json'
require 'open3'
require 'rbconfig'
require 'tmpdir'

ROOT = File.expand_path('..', __dir__)
SCREEN = File.join(ROOT, 'Scripts/laguna-authorized-screen.rb')
VECTOR = File.join(ROOT, 'Scripts/laguna-reviewed-vector.rb')
BROAD_REQUEST_TYPES = %w[
  bounded_automation
  bounded_discovery
  evidence_and_falsifier
  mechanism_analysis
  minimal_reproduction
  remediation_retest
].freeze
BROAD_CATEGORY_COUNT = 5

def assert(condition, message)
  raise message unless condition
end

def write_executable(path, body)
  File.write(path, "#!#{RbConfig.ruby}\n#{body}")
  File.chmod(0o700, path)
end

FAKE_SERVER = <<~'RUBY'
  require 'json'
  require 'socket'

  def value_after(name)
    index = ARGV.index(name)
    abort "missing #{name}" unless index
    ARGV.fetch(index + 1)
  end

  port = Integer(value_after('--port'))
  token = value_after('--api-key')
  log_path = File.join(File.dirname(__FILE__), 'requests.jsonl')
  server = TCPServer.new('127.0.0.1', port)
  trap('TERM') { server.close rescue nil; exit }
  loop do
    socket = server.accept
    request_line = socket.gets
    next socket.close unless request_line
    headers = {}
    while (line = socket.gets)
      break if line == "\r\n"
      key, value = line.split(':', 2)
      headers[key.downcase] = value.to_s.strip
    end
    body = socket.read(Integer(headers.fetch('content-length', '0')))
    _method, path, = request_line.split(' ')
    File.open(log_path, 'a', 0o600) { |file| file.puts(JSON.generate(path: path, body: body)) }
    code = 200
    payload = if headers['authorization'] != "Bearer #{token}"
                code = 401
                JSON.generate(error: 'unauthorized')
              elsif path == '/health'
                JSON.generate(status: 'ok')
              elsif path == '/apply-template'
                prompt = JSON.parse(body).fetch('messages').fetch(0).fetch('content')
                JSON.generate(prompt: "<system>Poolside fixture</system>\n<user>#{prompt}</user>\n<assistant><think>")
              elsif path == '/tokenize'
                content = JSON.parse(body).fetch('content')
                tokens = content.bytes.each_slice(8).map { |bytes| bytes.sum + 1 }
                tokens << 32_000
                JSON.generate(tokens: tokens)
              elsif path == '/v1/chat/completions'
                prompt = JSON.parse(body).fetch('messages').fetch(0).fetch('content')
                finish = prompt.include?('LENGTH_CASE') ? 'length' : 'stop'
                JSON.generate(choices: [{ finish_reason: finish,
                                          message: { content: "Substantive answer for #{prompt}",
                                                     reasoning_content: 'I cannot help appears only in hidden reasoning.' } }])
              else
                code = 404
                JSON.generate(error: 'not found')
              end
    reason = code == 200 ? 'OK' : (code == 401 ? 'Unauthorized' : 'Not Found')
    socket.write("HTTP/1.1 #{code} #{reason}\r\nContent-Type: application/json\r\nContent-Length: #{payload.bytesize}\r\nConnection: close\r\n\r\n#{payload}")
    socket.close
  end
RUBY

FAKE_GENERATOR = <<~'RUBY'
  require 'json'

  def value_after(name)
    index = ARGV.index(name)
    abort "missing #{name}" unless index
    ARGV.fetch(index + 1)
  end

  output = value_after('-o')
  File.write(File.join(File.dirname(output), 'generator-arguments.json'), JSON.generate(ARGV))
  File.binwrite(output, 'GGUF' + [3, 39, 3].pack('VQ<Q<'))
RUBY

def dataset_pair(name:, category:, request_type:, control:)
  {
    'name' => name, 'category' => category, 'requestType' => request_type, 'split' => 'train',
    'control' => control, 'contrast' => "NEVER_SEND #{name}",
    'controlSource' => "counterfactual-rewrite/task-specific-synthetic-range-v2/#{category};source=#{name}"
  }
end

def manifest_for(pairs)
  {
    'artifact_role' => 'counterfactual-control-candidates-not-screened',
    'asserted_input_split' => 'train',
    'derivations' => pairs.map do |pair|
      {
        'output_name' => pair.fetch('name'), 'equivalence' => 'same_operation_synthetic_target',
        'category' => pair.fetch('category'), 'request_type' => pair.fetch('requestType'),
        'generated_control_sha256' => Digest::SHA256.hexdigest(pair.fetch('control'))
      }
    end
  }
end

def decode_generator_line(line)
  output = +''
  index = 0
  while index < line.length
    if line[index] == '\\'
      index += 1
      raise 'dangling generator escape' if index >= line.length
      output << { '\\' => '\\', 'n' => "\n", 'r' => "\r", 't' => "\t" }.fetch(line[index], line[index])
    else
      output << line[index]
    end
    index += 1
  end
  output
end

Dir.mktmpdir('laguna-screen-worker-') do |temporary|
  server = File.join(temporary, 'fake-server')
  model = File.join(temporary, 'model.gguf')
  dataset_path = File.join(temporary, 'candidates.json')
  manifest_path = File.join(temporary, 'candidates.manifest.json')
  output = File.join(temporary, 'screen')
  write_executable(server, FAKE_SERVER)
  File.binwrite(model, 'model')
  pairs = 40.times.flat_map do |category_index|
    category = format('category-%02d', category_index)
    %w[b a].map do |suffix|
      control = "CONTROL #{category} #{suffix}"
      control += ' LENGTH_CASE' if category_index.zero? && suffix == 'a'
      dataset_pair(name: "#{category}-#{suffix}", category: category,
                   request_type: 'technical_review', control: control)
    end
  end
  File.write(dataset_path, JSON.generate(schema_version: 2, pairs: pairs))
  File.write(manifest_path, JSON.generate(manifest_for(pairs)))
  stdout, stderr, status = Open3.capture3(RbConfig.ruby, SCREEN, server, model, dataset_path,
                                          manifest_path, output, 'balanced')
  assert(status.success?, "authorized screen failed: #{stderr}\n#{stdout}")
  summary = JSON.parse(stdout)
  assert(summary['requests'] == 80 && summary['visible_answers'] == 80, 'screen counts are wrong')
  assert(summary['stopped'] == 79 && summary['length_limited'] == 1, 'finish reasons were conflated')
  assert(summary['refusal_candidates'].empty?, 'hidden reasoning was treated as a visible refusal')
  private_results = JSON.parse(File.binread(summary.dig('private_results', 'path')))
  expected_ids = pairs.group_by { |row| row.fetch('category') }.keys.sort.flat_map do |category|
    pairs.select { |row| row['category'] == category }.sort_by { |row| row['name'] }.map { |row| row['name'] }
  end
  assert(private_results.fetch('results').map { |row| row.fetch('id') } == expected_ids,
         'screen selection depends on input file order')
  requests = File.readlines(File.join(temporary, 'requests.jsonl'), chomp: true).map { |line| JSON.parse(line) }
  chats = requests.select { |row| row['path'] == '/v1/chat/completions' }.map { |row| JSON.parse(row['body']) }
  assert(chats.length == 80, 'screen did not send every selected prompt')
  assert(chats.all? { |body| body.dig('messages', 0, 'content').start_with?('CONTROL ') },
         'screen sent an excluded contrast')
  request_count = requests.length
  second_stdout, second_stderr, second_status = Open3.capture3(
    RbConfig.ruby, SCREEN, server, model, dataset_path, manifest_path, output, 'balanced'
  )
  assert(second_status.success?, "completed screen did not resume: #{second_stderr}\n#{second_stdout}")
  assert(File.readlines(File.join(temporary, 'requests.jsonl')).length == request_count,
         'completed screen unnecessarily restarted inference')
  checkpoint_path = File.join(output, 'checkpoint.json')
  checkpoint = JSON.parse(File.binread(checkpoint_path))
  checkpoint.fetch('results').first['content'] = 'forged checkpoint content'
  File.write(checkpoint_path, JSON.generate(checkpoint))
  _tampered_out, tampered_error, tampered_status = Open3.capture3(
    RbConfig.ruby, SCREEN, server, model, dataset_path, manifest_path, output, 'balanced'
  )
  assert(!tampered_status.success? && tampered_error.include?('extracted response differs'),
         'tampered checkpoint was published without inference')
  puts 'PASS: authorized screening binds provenance, orders controls, checkpoints, and separates visible refusal from completion'
end

Dir.mktmpdir('laguna-broad-screen-worker-') do |temporary|
  server = File.join(temporary, 'fake-server')
  model = File.join(temporary, 'model.gguf')
  dataset_path = File.join(temporary, 'candidates.json')
  manifest_path = File.join(temporary, 'candidates.manifest.json')
  pairable_output = File.join(temporary, 'pairable-screen')
  full_output = File.join(temporary, 'full-screen')
  write_executable(server, FAKE_SERVER)
  File.binwrite(model, 'model')
  pairs = BROAD_CATEGORY_COUNT.times.flat_map do |category_index|
    category = format('category-%02d', category_index)
    BROAD_REQUEST_TYPES.flat_map do |request_type|
      7.times.map do |row_index|
        dataset_pair(
          name: "#{category}-#{request_type}-#{row_index}", category: category,
          request_type: request_type,
          control: "CONTROL #{category} #{request_type} #{row_index}"
        )
      end
    end
  end.reverse
  File.write(dataset_path, JSON.generate(schema_version: 2, pairs: pairs))
  broad_manifest = manifest_for(pairs).merge(
    'coverage_requirements' => {
      'categories' => pairs.map { |row| row.fetch('category') }.uniq.reverse,
      'request_types' => BROAD_REQUEST_TYPES.reverse,
      'rows_per_cell' => 7
    }
  )
  File.write(manifest_path, JSON.generate(broad_manifest))

  cells = pairs.group_by { |row| [row.fetch('category'), row.fetch('requestType')] }
  full_ids = BROAD_CATEGORY_COUNT.times.flat_map do |category_index|
    category = format('category-%02d', category_index)
    BROAD_REQUEST_TYPES.flat_map do |request_type|
      cells.fetch([category, request_type]).sort_by { |row| row.fetch('name') }.map { |row| row.fetch('name') }
    end
  end
  pairable_ids = BROAD_CATEGORY_COUNT.times.flat_map do |category_index|
    category = format('category-%02d', category_index)
    BROAD_REQUEST_TYPES.flat_map do |request_type|
      cells.fetch([category, request_type]).sort_by { |row| row.fetch('name') }.first(2).map { |row| row.fetch('name') }
    end
  end

  stdout, stderr, status = Open3.capture3(
    RbConfig.ruby, SCREEN, server, model, dataset_path, manifest_path, pairable_output, 'pairable'
  )
  assert(status.success?, "pairable screen failed: #{stderr}\n#{stdout}")
  summary = JSON.parse(stdout)
  assert(summary['requests'] == 60 && summary['categories'] == BROAD_CATEGORY_COUNT &&
         summary['request_types'] == 6,
         'pairable screen coverage counts are wrong')
  pairable_results = JSON.parse(File.binread(summary.dig('private_results', 'path'))).fetch('results')
  assert(pairable_results.map { |row| row.fetch('id') } == pairable_ids,
         'pairable selection is not stable by category, request type, and record ID')

  pair_by_name = pairs.to_h { |pair| [pair.fetch('name'), pair] }
  full_results = full_ids.map do |id|
    pair = pair_by_name.fetch(id)
    content = "Substantive answer for #{pair.fetch('control')}"
    raw_response = JSON.generate(
      choices: [{ finish_reason: 'stop', message: { content: content, reasoning_content: '' } }]
    )
    {
      'id' => id, 'category' => pair.fetch('category'), 'request_type' => pair.fetch('requestType'),
      'split' => 'train', 'authorization' => 'explicit_owned_synthetic_fixture',
      'prompt_sha256' => Digest::SHA256.hexdigest(pair.fetch('control')),
      'rendered_prompt_sha256' => Digest::SHA256.hexdigest("rendered #{pair.fetch('control')}"),
      'raw_response_sha256' => Digest::SHA256.hexdigest(raw_response), 'raw_response' => raw_response,
      'http_status' => 200, 'content' => content, 'reasoning' => '', 'refusal_candidate' => false,
      'visible_answer' => true, 'finish_reason' => 'stop'
    }
  end
  full_configuration = {
    'mode' => 'full', 'reasoning_budget' => 128, 'max_tokens' => 768,
    'temperature' => 0, 'seed' => 42, 'selected_ids' => full_ids,
    'server_sha256' => Digest::SHA256.file(server).hexdigest,
    'model_sha256' => Digest::SHA256.file(model).hexdigest,
    'dataset_sha256' => Digest::SHA256.file(dataset_path).hexdigest,
    'manifest_sha256' => Digest::SHA256.file(manifest_path).hexdigest
  }
  FileUtils.mkdir_p(full_output)
  File.write(
    File.join(full_output, 'checkpoint.json'),
    JSON.generate(schema_version: 2, configuration: full_configuration, results: full_results)
  )
  request_count = File.readlines(File.join(temporary, 'requests.jsonl')).length
  full_stdout, full_stderr, full_status = Open3.capture3(
    RbConfig.ruby, SCREEN, server, model, dataset_path, manifest_path, full_output, 'full'
  )
  assert(full_status.success?, "full screen checkpoint validation failed: #{full_stderr}\n#{full_stdout}")
  full_summary = JSON.parse(full_stdout)
  assert(full_summary['requests'] == 210 && full_summary['categories'] == BROAD_CATEGORY_COUNT &&
         full_summary['request_types'] == 6, 'full screen coverage counts are wrong')
  assert(File.readlines(File.join(temporary, 'requests.jsonl')).length == request_count,
         'completed full screen checkpoint unnecessarily restarted inference')

  incomplete_pairs = pairs.reject.with_index { |_row, index| index.zero? }
  incomplete_dataset = File.join(temporary, 'incomplete.json')
  incomplete_manifest = File.join(temporary, 'incomplete.manifest.json')
  File.write(incomplete_dataset, JSON.generate(schema_version: 2, pairs: incomplete_pairs))
  incomplete_requirements = broad_manifest.fetch('coverage_requirements')
  File.write(
    incomplete_manifest,
    JSON.generate(manifest_for(incomplete_pairs).merge(
      'coverage_requirements' => incomplete_requirements))
  )
  _bad_out, bad_error, bad_status = Open3.capture3(
    RbConfig.ruby, SCREEN, server, model, incomplete_dataset, incomplete_manifest,
    File.join(temporary, 'incomplete-screen'), 'pairable'
  )
  assert(!bad_status.success? && bad_error.include?('expected 7 rows'),
         'pairable screening accepted an uneven category/request-type matrix')
  puts 'PASS: broad screening enforces and deterministically selects the complete category/request-type matrix'
end

Dir.mktmpdir('laguna-vector-worker-') do |temporary|
  server = File.join(temporary, 'fake-server')
  generator = File.join(temporary, 'fake-generator')
  model = File.join(temporary, 'model.gguf')
  dataset_path = File.join(temporary, 'candidates.json')
  manifest_path = File.join(temporary, 'candidates.manifest.json')
  screen_path = File.join(temporary, 'screen.json')
  selection_path = File.join(temporary, 'selection.json')
  output = File.join(temporary, 'vector')
  write_executable(server, FAKE_SERVER)
  write_executable(generator, FAKE_GENERATOR)
  File.binwrite(model, 'model')
  controls = [
    "owned positive one literal \\n and C:\\tmp\\x\nreal line\tend",
    "owned negative one literal \\t and C:\\tmp\\y\nreal line\tend",
    "owned positive two regex \\d+ and slash \\\\ value\nline",
    "owned negative two regex \\w+ and slash \\\\ value\nline"
  ]
  ids = %w[positive-1 negative-1 positive-2 negative-2]
  pairs = ids.zip(controls).map do |id, control|
    dataset_pair(name: id, category: 'oast-correlated-ssrf',
                 request_type: 'technical_review', control: control)
  end
  File.write(dataset_path, JSON.generate(schema_version: 2, pairs: pairs))
  File.write(manifest_path, JSON.generate(manifest_for(pairs)))
  render = lambda { |prompt| "<system>Poolside fixture</system>\n<user>#{prompt}</user>\n<assistant><think>" }
  configuration = {
    'mode' => 'ssrf', 'reasoning_budget' => 128, 'max_tokens' => 768,
    'temperature' => 0, 'seed' => 42, 'selected_ids' => ids,
    'server_sha256' => Digest::SHA256.file(server).hexdigest,
    'model_sha256' => Digest::SHA256.file(model).hexdigest,
    'dataset_sha256' => Digest::SHA256.file(dataset_path).hexdigest,
    'manifest_sha256' => Digest::SHA256.file(manifest_path).hexdigest
  }
  results = pairs.map.with_index do |pair, index|
    content = index.even? ? 'I cannot help with that.' : 'Here is the complete authorized answer.'
    raw_response = JSON.generate(choices: [{ finish_reason: 'stop',
                                             message: { content: content, reasoning_content: '' } }])
    {
      'id' => pair['name'], 'category' => pair['category'], 'request_type' => pair['requestType'],
      'split' => 'train', 'authorization' => 'explicit_owned_synthetic_fixture',
      'prompt_sha256' => Digest::SHA256.hexdigest(pair['control']),
      'rendered_prompt_sha256' => Digest::SHA256.hexdigest(render.call(pair['control'])),
      'raw_response_sha256' => Digest::SHA256.hexdigest(raw_response),
      'raw_response' => raw_response, 'http_status' => 200,
      'content' => content,
      'reasoning' => '', 'refusal_candidate' => index.even?, 'visible_answer' => true,
      'finish_reason' => 'stop'
    }
  end
  File.write(screen_path, JSON.generate(schema_version: 2, configuration: configuration, results: results))
  selection = {
    'schema_version' => 2, 'review_status' => 'semantically_reviewed',
    'reviewer' => { 'id' => 'reviewer-fixture', 'kind' => 'human',
                    'completed_at' => '2026-09-05T00:00:00Z',
                    'rubric_sha256' => Digest::SHA256.hexdigest('rubric') },
    'review_rubric' => 'rubric',
    'screen_results_sha256' => Digest::SHA256.file(screen_path).hexdigest,
    'dataset_sha256' => Digest::SHA256.file(dataset_path).hexdigest,
    'manifest_sha256' => Digest::SHA256.file(manifest_path).hexdigest,
    'model_sha256' => Digest::SHA256.file(model).hexdigest,
    'server_sha256' => Digest::SHA256.file(server).hexdigest,
    'records' => [[0, 1], [2, 3]].map.with_index do |(positive, negative), index|
      {
        'id' => "pair-#{index + 1}", 'positive_id' => ids[positive], 'negative_id' => ids[negative],
        'positive_response_sha256' => results[positive]['raw_response_sha256'],
        'negative_response_sha256' => results[negative]['raw_response_sha256'],
        'positive_outcome' => 'false_refusal', 'negative_outcome' => 'substantive_compliance',
        'equivalence_review' => 'same_task_authorized',
        'review_notes' => "notes #{index}",
        'review_notes_sha256' => Digest::SHA256.hexdigest("notes #{index}")
      }
    end
  }
  File.write(selection_path, JSON.generate(selection))
  rebind_server = lambda do |bound_server, label|
    bound_screen_path = File.join(temporary, "#{label}-screen.json")
    bound_selection_path = File.join(temporary, "#{label}-selection.json")
    bound_server_sha256 = Digest::SHA256.file(bound_server).hexdigest
    bound_screen = {
      'schema_version' => 2,
      'configuration' => configuration.merge('server_sha256' => bound_server_sha256),
      'results' => results
    }
    File.write(bound_screen_path, JSON.generate(bound_screen))
    bound_selection = selection.merge(
      'screen_results_sha256' => Digest::SHA256.file(bound_screen_path).hexdigest,
      'server_sha256' => bound_server_sha256
    )
    File.write(bound_selection_path, JSON.generate(bound_selection))
    [bound_screen_path, bound_selection_path]
  end
  stdout, stderr, status = Open3.capture3(RbConfig.ruby, VECTOR, selection_path, screen_path,
                                          dataset_path, manifest_path, model, server, generator, output)
  assert(status.success?, "reviewed vector failed: #{stderr}\n#{stdout}")
  summary = JSON.parse(stdout)
  assert(summary['pair_count'] == 2 && summary['vector']['bytes'] == 24 &&
         summary['runtime_load_verified'] == true, 'vector manifest is incomplete')
  algorithm = summary.fetch('algorithm')
  assert(!summary.key?('method') &&
         algorithm.dig('activation_capture', 'token_position') == 'final_rendered_prompt_token' &&
         algorithm.dig('direction', 'subtraction') == 'positive_minus_negative' &&
         algorithm.dig('direction', 'ordered_reduction') ==
           %w[per_layer_arithmetic_mean per_layer_l2_normalization] &&
         algorithm.dig('generator', 'cli_method') == 'mean' &&
         algorithm.dig('generator', 'executable_sha256') == Digest::SHA256.file(generator).hexdigest &&
         algorithm.dig('runtime_application', 'operation') == 'subtract_direction' &&
         algorithm.dig('runtime_application', 'control_vector_scaled_argument') == -0.25,
         'vector algorithm provenance is vague or incomplete')
  tokenization = summary.fetch('tokenization')
  assert(tokenization['endpoint'] == '/tokenize' &&
         tokenization['generation_boundary_token_id'] == 32_000 &&
         tokenization.fetch('pairs').map { |row| row.fetch('id') } == %w[pair-1 pair-2] &&
         tokenization.fetch('pairs').all? do |row|
           row.fetch('positive_token_count').positive? && row.fetch('negative_token_count').positive? &&
             row.fetch('positive_to_negative_ratio').between?(0.8, 1.25)
         end,
         'vector tokenization provenance is incomplete')
  positive_lines = File.readlines(File.join(output, 'positive-rendered.txt'), chomp: true).map { |line| decode_generator_line(line) }
  negative_lines = File.readlines(File.join(output, 'negative-rendered.txt'), chomp: true).map { |line| decode_generator_line(line) }
  assert(positive_lines == [render.call(controls[0]), render.call(controls[2])], 'positive prompt escaping changed bytes')
  assert(negative_lines == [render.call(controls[1]), render.call(controls[3])], 'negative prompt escaping changed bytes')
  arguments = JSON.parse(File.read(File.join(output, 'generator-arguments.json')))
  assert(arguments.each_cons(2).any? { |a, b| a == '--method' && b == 'mean' }, 'generator method drifted')
  requests = File.readlines(File.join(temporary, 'requests.jsonl'), chomp: true).map { |line| JSON.parse(line) }
  tokenize_requests = requests.select { |row| row['path'] == '/tokenize' }.map { |row| JSON.parse(row['body']) }
  assert(tokenize_requests.map { |request| request.fetch('content') } == controls.map { |control| render.call(control) } &&
         tokenize_requests.all? do |request|
           request['add_special'] == true && request['parse_special'] == true &&
             request['with_pieces'] == false
         end,
         'vector worker did not tokenize every exact rendered prompt with fixed options')

  rejected = selection.merge('review_status' => 'pending')
  rejected_path = File.join(temporary, 'rejected-selection.json')
  File.write(rejected_path, JSON.generate(rejected))
  _out, rejected_error, rejected_status = Open3.capture3(
    RbConfig.ruby, VECTOR, rejected_path, screen_path, dataset_path, manifest_path,
    model, server, generator, File.join(temporary, 'rejected-output')
  )
  assert(!rejected_status.success? && rejected_error.include?('semantic review'),
         'unreviewed selection reached vector generation')

  invalid_generator = File.join(temporary, 'invalid-generator')
  write_executable(invalid_generator, FAKE_GENERATOR.sub(
    "File.binwrite(output, 'GGUF' + [3, 39, 3].pack('VQ<Q<'))",
    "File.binwrite(output, 'not-a-vector')"
  ))
  _invalid_out, invalid_error, invalid_status = Open3.capture3(
    RbConfig.ruby, VECTOR, selection_path, screen_path, dataset_path, manifest_path,
    model, server, invalid_generator, File.join(temporary, 'invalid-vector-output')
  )
  assert(!invalid_status.success? && invalid_error.include?('not a GGUF'),
         'non-GGUF generator output was accepted as a vector')

  malformed_server = File.join(temporary, 'malformed-tokenize-server')
  write_executable(malformed_server, FAKE_SERVER.sub('JSON.generate(tokens: tokens)',
                                                     "JSON.generate(tokens: [])"))
  malformed_screen_path, malformed_selection_path = rebind_server.call(malformed_server, 'malformed-tokenize')
  _malformed_out, malformed_error, malformed_status = Open3.capture3(
    RbConfig.ruby, VECTOR, malformed_selection_path, malformed_screen_path, dataset_path, manifest_path,
    model, malformed_server, generator, File.join(temporary, 'malformed-tokenize-output')
  )
  assert(!malformed_status.success? && malformed_error.include?('nonempty integer token IDs'),
         'malformed tokenizer output reached vector generation')

  mismatched_server = File.join(temporary, 'mismatched-boundary-server')
  write_executable(mismatched_server, FAKE_SERVER.sub(
    'tokens << 32_000', "tokens << (content.include?('owned positive') ? 32_001 : 32_000)"
  ))
  mismatched_screen_path, mismatched_selection_path = rebind_server.call(mismatched_server, 'mismatched-boundary')
  _mismatched_out, mismatched_error, mismatched_status = Open3.capture3(
    RbConfig.ruby, VECTOR, mismatched_selection_path, mismatched_screen_path, dataset_path, manifest_path,
    model, mismatched_server, generator, File.join(temporary, 'mismatched-boundary-output')
  )
  assert(!mismatched_status.success? && mismatched_error.include?('final generation-boundary token'),
         'mismatched final prompt tokens reached vector generation')

  unbalanced_server = File.join(temporary, 'unbalanced-token-server')
  write_executable(unbalanced_server, FAKE_SERVER.sub(
    'tokens = content.bytes.each_slice(8).map { |bytes| bytes.sum + 1 }',
    "tokens = Array.new(content.include?('owned positive') ? 100 : 2, 1)"
  ))
  unbalanced_screen_path, unbalanced_selection_path = rebind_server.call(unbalanced_server, 'unbalanced-token')
  _unbalanced_out, unbalanced_error, unbalanced_status = Open3.capture3(
    RbConfig.ruby, VECTOR, unbalanced_selection_path, unbalanced_screen_path, dataset_path, manifest_path,
    model, unbalanced_server, generator, File.join(temporary, 'unbalanced-token-output')
  )
  assert(!unbalanced_status.success? && unbalanced_error.include?('token-length balance'),
         'token-length-imbalanced prompts reached vector generation')
  puts 'PASS: reviewed vectors bind exact rendered tokens, selection evidence, and explicit algorithm provenance'
end
