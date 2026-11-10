require_relative "test_helper"
require "transformers-rb"

# Issue #32: from_pretrained(labels:) saves real label names
class LabelsTest < Minitest::Test
  def setup
    @tmpdir = Dir.mktmpdir
    @requested = []
    requested = @requested
    # No network: build a tiny DistilBERT with the num_labels asked for
    stub(Transformers::AutoModelForSequenceClassification, :from_pretrained) do |_name, num_labels:|
      requested << num_labels
      Transformers::Distilbert::DistilBertForSequenceClassification.new(
        Transformers::Distilbert::DistilBertConfig.new(vocab_size: 8, dim: 8, n_layers: 1, n_heads: 2,
                                                       hidden_dim: 16, max_position_embeddings: 16,
                                                       num_labels: num_labels)
      )
    end
    stub(Transformers::AutoTokenizer, :from_pretrained) { |_name| nil }
  end

  def teardown
    @stubs.each do |mod, name, original|
      mod.singleton_class.remove_method(name)
      mod.define_singleton_method(name, original) if original
    end
    FileUtils.rm_rf(@tmpdir)
  end

  def test_labels_set_the_config_and_survive_save_and_load
    model, = Trainers.from_pretrained("tiny", labels: %w[negative neutral positive])
    assert_equal [3], @requested
    assert_equal({ 0 => "negative", 1 => "neutral", 2 => "positive" }, model.config.id2label)
    assert_equal 3, model.config.num_labels

    quietly { Trainers::SaveUtils.save_pretrained(model, nil, @tmpdir) }
    config = JSON.parse(File.read(File.join(@tmpdir, "config.json")))
    assert_equal({ "0" => "negative", "1" => "neutral", "2" => "positive" }, config["id2label"])
    assert_equal({ "negative" => 0, "neutral" => 1, "positive" => 2 }, config["label2id"])

    reloaded, = Trainers.load_pretrained(@tmpdir)
    assert_equal model.config.id2label, reloaded.config.id2label
  end

  def test_symbols_become_strings
    model, = Trainers.from_pretrained("tiny", labels: %i[neg pos], num_labels: 2)
    assert_equal({ 0 => "neg", 1 => "pos" }, model.config.id2label)
  end

  def test_num_labels_alone_still_works
    Trainers.from_pretrained("tiny")
    Trainers.from_pretrained("tiny", num_labels: 5)
    assert_equal [2, 5], @requested
  end

  def test_bad_labels_raise
    error = assert_raises(ArgumentError) { Trainers.from_pretrained("tiny", labels: %w[a b c], num_labels: 2) }
    assert_match "num_labels: 2 contradicts labels", error.message
    assert_raises(ArgumentError) { Trainers.from_pretrained("tiny", labels: []) }
    assert_raises(ArgumentError) { Trainers.from_pretrained("tiny", labels: %w[a a]) }
    assert_empty @requested
  end

  private

  def stub(mod, name, &block)
    # An inherited class method is restored by removing the stub
    own = mod.singleton_class.instance_methods(false).include?(name)
    (@stubs ||= []) << [mod, name, own ? mod.method(name) : nil]
    mod.define_singleton_method(name, &block)
  end
end
