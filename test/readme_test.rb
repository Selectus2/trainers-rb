require_relative "test_helper"
require "transformers-rb"

# Issue #35: the README's architectures table is checked against what
# transformers-rb provides, so it cannot drift from the gem
class ReadmeTest < Minitest::Test
  README = File.expand_path("../README.md", __dir__)
  TASKS = {
    "Sequence classification" => Transformers::MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES,
    "Token classification"    => Transformers::MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES,
    "Question answering"      => Transformers::MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES
  }.freeze
  # Heads that exist in transformers-rb but cannot be trained
  UNTRAINABLE = { "Question answering" => "no loss" }.freeze

  def test_architectures_table_matches_transformers_rb
    readme = File.read(README)
    table  = readme[/<!-- architectures:start -->\n(.*?)<!-- architectures:end -->/m, 1]
    refute_nil table, "README is missing the architectures table markers"
    assert_equal expected_table, table, "Update the README architectures table to:\n\n#{expected_table}"
  end

  private

  def expected_table
    types = Trainers::LoraConfig::DEFAULT_TARGETS.keys | TASKS.values.flat_map(&:keys)
    types = types.select { |type| Transformers::CONFIG_MAPPING_NAMES.key?(type) }.sort

    rows = types.map do |type|
      cells = TASKS.map do |task, mapping|
        next "–" unless mapping.key?(type)

        UNTRAINABLE[task] ? "✗ #{UNTRAINABLE[task]}" : "✓"
      end
      targets = Trainers::LoraConfig::DEFAULT_TARGETS.fetch(type, []).map { |name| "`#{name}`" }.join(", ")
      "| `#{type}` | #{cells.join(' | ')} | #{targets} |"
    end

    header = "| `model_type` | #{TASKS.keys.join(' | ')} | Default LoRA targets |"
    divider = "|#{'---|' * (TASKS.size + 2)}"
    ([header, divider] + rows).join("\n") + "\n"
  end
end
