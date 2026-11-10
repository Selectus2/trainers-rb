require_relative "test_helper"
require "open3"
require "rbconfig"
require "transformers-rb"

# Issue #17: everything save_model writes can be read back
class SaveLoadTest < Minitest::Test
  TEXTS = ["great film", "awful film", "loved it", "hated it"].freeze
  VOCAB = %w[[PAD] [UNK] [CLS] [SEP] [MASK] great awful film loved hated it what a].each_with_index.to_h.freeze

  def setup
    Trainers.set_seed(0)
    @tmpdir = Dir.mktmpdir
    @tokenizer = build_tokenizer
    @model = build_model
  end

  def teardown
    FileUtils.rm_rf(@tmpdir)
  end

  def test_trained_model_reloads_with_identical_predictions_in_a_fresh_process
    train_two_steps
    model_dir = File.join(@tmpdir, "model")
    quietly { @trainer.save_model(model_dir) }

    @model.eval
    encoded  = @tokenizer.(["what a great film"], return_tensors: "pt")
    expected = @model.(input_ids: encoded[:input_ids], attention_mask: encoded[:attention_mask])[:logits].to_a

    script = <<~RUBY
      require "trainers-rb"
      Trainers.logger = nil
      model, tokenizer = Trainers.load_pretrained(ARGV[0])
      encoded = tokenizer.(["what a great film"], return_tensors: "pt")
      logits = model.(input_ids: encoded[:input_ids], attention_mask: encoded[:attention_mask])[:logits]
      puts Marshal.dump([model.class.name, tokenizer.class.name, model.config.id2label, logits.to_a]).unpack1("H*")
    RUBY
    lib = File.expand_path("../lib", __dir__)
    out, err, status = Open3.capture3(RbConfig.ruby, "-I", lib, "-e", script, model_dir)
    assert status.success?, err

    model_class, tokenizer_class, id2label, logits = Marshal.load([out.lines.last.strip].pack("H*"))
    assert_equal "Transformers::Distilbert::DistilBertForSequenceClassification", model_class
    assert_equal "Transformers::Distilbert::DistilBertTokenizerFast", tokenizer_class
    assert_equal({ 0 => "NEG", 1 => "POS" }, id2label)
    assert_equal expected, logits
  end

  def test_directory_contents
    quietly { Trainers::SaveUtils.save_pretrained(@model, @tokenizer, @tmpdir) }

    assert_equal %w[config.json model.safetensors tokenizer.json tokenizer_config.json], Dir.children(@tmpdir).sort

    config = JSON.parse(File.read(File.join(@tmpdir, "config.json")))
    assert_equal ["DistilBertForSequenceClassification"], config["architectures"]
    assert_equal "distilbert", config["model_type"]
    assert_equal({ "0" => "NEG", "1" => "POS" }, config["id2label"])
    assert_equal({ "NEG" => 0, "POS" => 1 }, config["label2id"])

    tokenizer_config = JSON.parse(File.read(File.join(@tmpdir, "tokenizer_config.json")))
    assert_equal "DistilBertTokenizerFast", tokenizer_config["tokenizer_class"]
    assert_equal "[PAD]", tokenizer_config["pad_token"]

    metadata = nil
    Safetensors.safe_open(File.join(@tmpdir, "model.safetensors"), framework: "pt") { |f| metadata = f.metadata }
    assert_equal "pt", metadata["format"]
  end

  def test_plain_tokenizers_tokenizer_round_trips
    backend = @tokenizer.backend_tokenizer
    quietly { Trainers::SaveUtils.save_pretrained(@model, backend, @tmpdir) }
    refute File.exist?(File.join(@tmpdir, "tokenizer_config.json"))

    _model, tokenizer = Trainers.load_pretrained(@tmpdir)
    assert_instance_of Tokenizers::Tokenizer, tokenizer
    assert_equal backend.encode("loved it").ids, tokenizer.encode("loved it").ids
  end

  def test_without_tokenizer_loads_nil_tokenizer
    quietly { Trainers::SaveUtils.save_pretrained(@model, nil, @tmpdir) }
    _model, tokenizer = Trainers.load_pretrained(@tmpdir)
    assert_nil tokenizer
  end

  def test_model_without_config_cannot_be_loaded
    quietly { Trainers::SaveUtils.save_pretrained(TinyClassifier.new, nil, @tmpdir) }
    refute File.exist?(File.join(@tmpdir, "config.json"))

    error = assert_raises(Trainers::CheckpointError) { Trainers.load_pretrained(@tmpdir) }
    assert_match "No config.json", error.message
  end

  def test_unmerged_lora_raises_adapter_error
    quietly { Trainers::LoraModel.apply(@model, Trainers::LoraConfig.new(r: 2, target_modules: ["q_lin"])) }
    quietly { Trainers::SaveUtils.save_pretrained(@model, nil, @tmpdir) }

    error = assert_raises(Trainers::AdapterError) { Trainers.load_pretrained(@tmpdir) }
    assert_match "LoraModel.merge", error.message
  end

  def test_merged_lora_round_trips
    quietly do
      Trainers::LoraModel.apply(@model, Trainers::LoraConfig.new(r: 2, target_modules: ["q_lin"]))
      Trainers::LoraModel.merge(@model)
      Trainers::SaveUtils.save_pretrained(@model, nil, @tmpdir)
    end

    model, _tokenizer = Trainers.load_pretrained(@tmpdir)
    ids = Torch.tensor([[2, 5, 7, 3]])
    mask = Torch.ones_like(ids)
    @model.eval
    assert_equal @model.(input_ids: ids, attention_mask: mask)[:logits].to_a,
                 model.(input_ids: ids, attention_mask: mask)[:logits].to_a
  end

  # A head built from a pretrained config that names another architecture,
  # such as DistilBertForMaskedLM, is saved under its own class
  def test_architectures_names_the_saved_class
    config = Transformers::Distilbert::DistilBertConfig.new(
      vocab_size: VOCAB.size, dim: 16, n_layers: 1, n_heads: 2, hidden_dim: 32, max_position_embeddings: 32,
      architectures: ["DistilBertForMaskedLM"]
    )
    model = Transformers::Distilbert::DistilBertForSequenceClassification.new(config)
    quietly { Trainers::SaveUtils.save_pretrained(model, nil, @tmpdir) }

    saved = JSON.parse(File.read(File.join(@tmpdir, "config.json")))
    assert_equal ["DistilBertForSequenceClassification"], saved["architectures"]
    assert_instance_of Transformers::Distilbert::DistilBertForSequenceClassification, Trainers.load_pretrained(@tmpdir).first
  end

  def test_mismatched_weights_raise_checkpoint_error
    quietly { Trainers::SaveUtils.save_pretrained(@model, nil, @tmpdir) }
    config_path = File.join(@tmpdir, "config.json")
    config = JSON.parse(File.read(config_path)).merge("dim" => 8, "hidden_dim" => 16)
    File.write(config_path, JSON.generate(config))

    assert_raises(Trainers::CheckpointError) { Trainers.load_pretrained(@tmpdir) }
  end

  private

  def build_tokenizer
    backend = Tokenizers::Tokenizer.new(Tokenizers::Models::WordLevel.new(vocab: VOCAB, unk_token: "[UNK]"))
    backend.pre_tokenizer = Tokenizers::PreTokenizers::Whitespace.new
    backend.post_processor = Tokenizers::Processors::TemplateProcessing.new(
      single: "[CLS] $A [SEP]", pair: "[CLS] $A [SEP] $B:1 [SEP]:1",
      special_tokens: [["[CLS]", VOCAB["[CLS]"]], ["[SEP]", VOCAB["[SEP]"]]]
    )
    path = File.join(@tmpdir, "source-tokenizer.json")
    backend.save(path)
    tokenizer = Transformers::Distilbert::DistilBertTokenizerFast.new(tokenizer_file: path)
    File.delete(path)
    tokenizer
  end

  def build_model
    config = Transformers::Distilbert::DistilBertConfig.new(
      vocab_size: VOCAB.size, dim: 16, n_layers: 1, n_heads: 2, hidden_dim: 32, max_position_embeddings: 32,
      num_labels: 2, id2label: { 0 => "NEG", 1 => "POS" }, label2id: { "NEG" => 0, "POS" => 1 }
    )
    Transformers::Distilbert::DistilBertForSequenceClassification.new(config)
  end

  def train_two_steps
    data = TEXTS.each_with_index.map do |text, i|
      encoded = @tokenizer.(text)
      { input_ids: encoded["input_ids"], attention_mask: encoded["attention_mask"], labels: (i + 1) % 2 }
    end
    args = Trainers::TrainingArguments.new(
      output_dir: File.join(@tmpdir, "run"), num_train_epochs: 1, per_device_train_batch_size: 2,
      save_strategy: :no, device: "cpu"
    )
    @trainer = Trainers::Trainer.new(
      model: @model, args: args, train_dataset: Trainers::Dataset.new(data), tokenizer: @tokenizer,
      data_collator: Trainers::DataCollatorWithPadding.new(tokenizer: @tokenizer)
    )
    quietly { @trainer.train }
  end
end
