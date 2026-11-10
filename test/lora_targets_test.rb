require_relative "test_helper"
require "transformers-rb"
require "stringio"

# A query projection next to a longer name that contains it, like DeBERTa's
# query_proj and pos_query_proj
class TinyRelativeAttention < Torch::NN::Module
  def initialize
    super()
    @query     = Torch::NN::Linear.new(4, 4)
    @pos_query = Torch::NN::Linear.new(4, 4)
    @value     = Torch::NN::Linear.new(4, 4)
  end

  def forward(x)
    @query.call(x) + @pos_query.call(x) + @value.call(x)
  end
end

# Issue #33: default target_modules per architecture, and PEFT-style matching
class LoraTargetsTest < Minitest::Test
  # transformers-rb class, config overrides, and the modules the defaults must adapt in layer 0
  ARCHITECTURES = {
    "bert"        => [Transformers::Bert::BertForSequenceClassification, Transformers::Bert::BertConfig, {},
                      %w[bert.encoder.layer.0.attention.self.query bert.encoder.layer.0.attention.self.value]],
    "xlm-roberta" => [Transformers::XlmRoberta::XLMRobertaForSequenceClassification, Transformers::XlmRoberta::XLMRobertaConfig, {},
                      %w[roberta.encoder.layer.0.attention.self.query roberta.encoder.layer.0.attention.self.value]],
    "distilbert"  => [Transformers::Distilbert::DistilBertForSequenceClassification, Transformers::Distilbert::DistilBertConfig,
                      { dim: 8, n_layers: 1, n_heads: 2, hidden_dim: 16 },
                      %w[distilbert.transformer.layer.0.attention.q_lin distilbert.transformer.layer.0.attention.v_lin]],
    "deberta-v2"  => [Transformers::DebertaV2::DebertaV2ForSequenceClassification, Transformers::DebertaV2::DebertaV2Config,
                      { relative_attention: true, pos_att_type: %w[p2c c2p] },
                      %w[deberta.encoder.layer.0.attention.self.query_proj deberta.encoder.layer.0.attention.self.value_proj]],
    "mpnet"       => [Transformers::Mpnet::MPNetForSequenceClassification, Transformers::Mpnet::MPNetConfig, {},
                      %w[mpnet.encoder.layer.0.attention.attn.q mpnet.encoder.layer.0.attention.attn.v]]
  }.freeze

  def setup
    @saved_logger = Trainers.logger
    @log = StringIO.new
    Trainers.logger = Logger.new(@log)
  end

  def teardown
    Trainers.logger = @saved_logger
  end

  def test_defaults_adapt_query_and_value_for_each_architecture
    ARCHITECTURES.each do |type, (model_class, config_class, overrides, expected)|
      config = config_class.new(vocab_size: 16, hidden_size: 8, num_hidden_layers: 1, num_attention_heads: 2,
                                intermediate_size: 16, max_position_embeddings: 16, num_labels: 2, **overrides)
      assert_equal type, config.model_type
      model = model_class.new(config)
      Trainers::LoraModel.apply(model, Trainers::LoraConfig.new(r: 2))

      assert_equal expected, lora_names(model), type
    end
    assert_match "(default)", @log.string
  end

  def test_unknown_model_type_without_targets_raises
    model = TinyLoraTarget.new
    config = Struct.new(:model_type).new("gpt2")
    model.define_singleton_method(:config) { config }

    error = assert_raises(Trainers::AdapterError) { Trainers::LoraModel.apply(model, Trainers::LoraConfig.new) }
    assert_match "No default target_modules for model_type \"gpt2\"", error.message
    assert_match "attn.query", error.message
  end

  def test_models_without_a_config_keep_the_old_default
    model = TinyLoraTarget.new
    Trainers::LoraModel.apply(model, Trainers::LoraConfig.new(r: 2))
    assert_equal %w[attn.query attn.value], lora_names(model)
  end

  def test_explicit_targets_win_over_the_defaults
    model = tiny_distilbert
    Trainers::LoraModel.apply(model, Trainers::LoraConfig.new(r: 2, target_modules: ["q_lin"]))
    assert_equal %w[distilbert.transformer.layer.0.attention.q_lin], lora_names(model)
  end

  def test_targets_match_whole_name_parts
    model = TinyRelativeAttention.new
    Trainers::LoraModel.apply(model, Trainers::LoraConfig.new(r: 2, target_modules: %w[query value]))
    assert_equal %w[query value], lora_names(model)
    refute_match "deprecated", @log.string
  end

  def test_full_names_match
    model = TinyLoraTarget.new
    Trainers::LoraModel.apply(model, Trainers::LoraConfig.new(r: 2, target_modules: ["attn.value"]))
    assert_equal %w[attn.value], lora_names(model)
  end

  def test_substring_targets_still_match_with_a_deprecation_warning
    model = TinyLoraTarget.new
    Trainers::LoraModel.apply(model, Trainers::LoraConfig.new(r: 2, target_modules: %w[que out]))
    assert_equal %w[attn.query out], lora_names(model)
    assert_match "target_modules \"que\" matched attn.query by substring", @log.string
    refute_match "\"out\" matched", @log.string, "an exact match must not warn"
  end

  private

  def lora_names(model)
    model.named_modules.select { |_, mod| mod.is_a?(Trainers::LoraLinear) }.map(&:first).grep_v(/layer\.[1-9]/)
  end

  def tiny_distilbert
    Transformers::Distilbert::DistilBertForSequenceClassification.new(
      Transformers::Distilbert::DistilBertConfig.new(vocab_size: 16, dim: 8, n_layers: 1, n_heads: 2, hidden_dim: 16,
                                                     max_position_embeddings: 16)
    )
  end
end
