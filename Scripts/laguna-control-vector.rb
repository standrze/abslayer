#!/usr/bin/env ruby
# frozen_string_literal: true

require 'digest'
require 'fileutils'
require 'json'
require 'open3'

abort 'usage: laguna-control-vector.rb DATASET MODEL GENERATOR OUTPUT COUNT' unless ARGV.length == 5
dataset_path, model_path, generator_path, output_path, count_text = ARGV
count = Integer(count_text, 10)
raise 'count must be between 2 and 128' unless (2..128).cover?(count)
[dataset_path, model_path, generator_path].each do |path|
  raise "missing regular input: #{path}" unless File.file?(path) && !File.symlink?(path)
end
raise 'output already exists' if File.exist?(output_path)

document = JSON.parse(File.binread(dataset_path), create_additions: false)
raise 'dataset schema must be promptfile-v2' unless document['schema_version'] == 2
pairs = document['pairs']
raise 'dataset pairs are missing' unless pairs.is_a?(Array) && pairs.length >= count
groups = pairs.group_by { |record| record.fetch('category') }
raise 'dataset needs at least two categories' unless groups.length >= 2

selected = []
offset = 0
categories = groups.keys.sort
while selected.length < count
  added = false
  categories.each do |category|
    record = groups.fetch(category)[offset]
    next unless record
    %w[name contrast control category].each do |key|
      raise "invalid #{key}" unless record[key].is_a?(String) && !record[key].strip.empty?
    end
    selected << record
    added = true
    break if selected.length == count
  end
  raise 'not enough category-balanced records' unless added
  offset += 1
end

FileUtils.mkdir_p(File.dirname(output_path), mode: 0o700)
Dir.mkdir(output_path, 0o700)
positive_path = File.join(output_path, 'positive.txt')
negative_path = File.join(output_path, 'negative.txt')
vector_path = File.join(output_path, 'control-vector.gguf')
escape_line = lambda do |text|
  text.gsub('\\', '\\\\').gsub("\n", '\\n').gsub("\r", '\\r').gsub("\t", '\\t')
end
File.open(positive_path, File::WRONLY | File::CREAT | File::EXCL, 0o600) do |file|
  selected.each { |record| file.puts(escape_line.call(record.fetch('contrast'))) }
end
File.open(negative_path, File::WRONLY | File::CREAT | File::EXCL, 0o600) do |file|
  selected.each { |record| file.puts(escape_line.call(record.fetch('control'))) }
end

arguments = [generator_path, '-m', model_path,
             '--positive-file', positive_path, '--negative-file', negative_path,
             '--method', 'mean', '--n-gpu-layers', 'all',
             '-c', '2048', '-b', '512', '-ub', '128', '-o', vector_path]
status = nil
Open3.popen3(*arguments) do |stdin, stdout, stderr, wait|
  stdin.close
  output_thread = Thread.new { stdout.each_line { |line| warn(line) } }
  error_thread = Thread.new { stderr.each_line { |line| warn(line) } }
  output_thread.join
  error_thread.join
  status = wait.value
end
raise "control-vector generator exited #{status.exitstatus}" unless status.success?
raise 'generator did not publish a vector' unless File.file?(vector_path) && File.size(vector_path).positive?

puts(JSON.generate(
  operation: 'laguna_control_vector', status: 'completed', method: 'mean_final_token',
  pair_count: selected.length,
  dataset: { path: dataset_path, sha256: Digest::SHA256.file(dataset_path).hexdigest },
  model: { path: model_path, sha256: Digest::SHA256.file(model_path).hexdigest },
  generator: { path: generator_path, sha256: Digest::SHA256.file(generator_path).hexdigest },
  vector: { path: vector_path, bytes: File.size(vector_path), sha256: Digest::SHA256.file(vector_path).hexdigest },
  acceptance: 'not_evaluated'
))
