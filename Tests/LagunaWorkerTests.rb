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
  stdout, stderr, status = Open3.capture3(RbConfig.ruby, VECTOR, selection_path, screen_path,
                                          dataset_path, manifest_path, model, server, generator, output)
  assert(status.success?, "reviewed vector failed: #{stderr}\n#{stdout}")
  summary = JSON.parse(stdout)
  assert(summary['pair_count'] == 2 && summary['vector']['bytes'] == 24 &&
         summary['runtime_load_verified'] == true, 'vector manifest is incomplete')
  positive_lines = File.readlines(File.join(output, 'positive-rendered.txt'), chomp: true).map { |line| decode_generator_line(line) }
  negative_lines = File.readlines(File.join(output, 'negative-rendered.txt'), chomp: true).map { |line| decode_generator_line(line) }
  assert(positive_lines == [render.call(controls[0]), render.call(controls[2])], 'positive prompt escaping changed bytes')
  assert(negative_lines == [render.call(controls[1]), render.call(controls[3])], 'negative prompt escaping changed bytes')
  arguments = JSON.parse(File.read(File.join(output, 'generator-arguments.json')))
  assert(arguments.each_cons(2).any? { |a, b| a == '--method' && b == 'mean' }, 'generator method drifted')

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
  puts 'PASS: reviewed vectors bind screen evidence and preserve exact escaped template bytes'
end
