require_relative "test_helper"

class ErrorsTest < Minitest::Test
  def setup
    @tmpdir = Dir.mktmpdir
  end

  def teardown
    FileUtils.rm_rf(@tmpdir)
  end

  def test_hierarchy
    assert_operator Trainers::Error, :<, StandardError
    %w[ConfigurationError AdapterError CheckpointError].each do |name|
      assert_operator Trainers.const_get(name), :<, Trainers::Error, name
    end
  end

  def test_save_lora_adapters_without_adapters_raises
    error = assert_raises(Trainers::AdapterError) do
      Trainers::SaveUtils.save_lora_adapters(TinyLoraTarget.new, @tmpdir)
    end
    assert_match "No LoRA adapters", error.message
    refute File.exist?(File.join(@tmpdir, "adapter_model.safetensors"))
  end

  def test_load_adapters_from_missing_dir_raises
    model = TinyLoraTarget.new
    Trainers::LoraModel.apply(model, Trainers::LoraConfig.new(r: 2))

    assert_raises(Trainers::AdapterError) do
      Trainers::LoraModel.load_adapters(model, File.join(@tmpdir, "nope"))
    end
  end

  def test_replace_module_with_bad_path_raises
    assert_raises(Trainers::AdapterError) do
      Trainers::LoraUtils.replace_module(TinyLoraTarget.new, "missing.query", Torch::NN::Linear.new(2, 2))
    end
  end

  def test_no_trainable_parameters_raises
    model = TinyClassifier.new
    model.parameters.each { |p| p.requires_grad = false }

    error = assert_raises(Trainers::ConfigurationError) do
      Trainers::Optimization.create_optimizer(model, training_args)
    end
    assert_match "No trainable parameters", error.message
  end

  def test_missing_labels_raises
    rows = Array.new(4) { { input: [0.1, 0.2, 0.3, 0.4] } }
    trainer = Trainers::Trainer.new(model: TinyClassifier.new, args: training_args(num_train_epochs: 1),
                                    train_dataset: Trainers::Dataset.new(rows))

    assert_raises(Trainers::ConfigurationError) { trainer.train }
  end

  def test_rescuing_the_base_class_catches_everything
    assert_raises(Trainers::Error) do
      Trainers::SaveUtils.save_lora_adapters(TinyLoraTarget.new, @tmpdir)
    end
  end
end
