require_relative "test_helper"

# Every README snippet, example and talk slide is a call site we cannot edit.
# This test pins the public surface so a rename or a dropped keyword fails CI.
class PublicApiTest < Minitest::Test
  MESSAGE = "public API changed — was this intentional? update this test and the CHANGELOG"

  def test_trainers_from_pretrained
    assert_params [[:req, :model_name], [:key, :task], [:key, :num_labels]],
                  Trainers.method(:from_pretrained)
  end

  def test_seed_and_logger
    assert_params [[:req, :seed]], Trainers.method(:set_seed)
    assert_respond_to Trainers, :logger, MESSAGE
    assert_respond_to Trainers, :logger=, MESSAGE
    assert_params [], Trainers.method(:default_logger)
  end

  def test_error_hierarchy
    %i[Error ConfigurationError AdapterError CheckpointError].each do |name|
      assert Trainers.const_defined?(name, false), "#{MESSAGE}: Trainers::#{name} is missing"
      assert_operator Trainers.const_get(name), :<=, Trainers::Error, MESSAGE
    end
  end

  def test_trainer
    assert_public_methods Trainers::Trainer,
                          %i[train evaluate predict save_model model args train_dataset eval_dataset
                             tokenizer data_collator optimizer lr_scheduler state control]

    assert_params [[:keyreq, :model], [:key, :args], [:key, :train_dataset], [:key, :eval_dataset],
                   [:key, :tokenizer], [:key, :data_collator], [:key, :compute_metrics], [:key, :callbacks]],
                  Trainers::Trainer.instance_method(:initialize)
    assert_params [], Trainers::Trainer.instance_method(:train)
    assert_params [[:key, :eval_dataset]], Trainers::Trainer.instance_method(:evaluate)
    assert_params [[:req, :test_dataset]], Trainers::Trainer.instance_method(:predict)
    assert_params [[:opt, :output_dir]], Trainers::Trainer.instance_method(:save_model)
  end

  def test_training_arguments
    expected_keys = %i[
      output_dir num_train_epochs per_device_train_batch_size per_device_eval_batch_size
      learning_rate weight_decay adam_beta1 adam_beta2 adam_epsilon max_grad_norm
      gradient_accumulation_steps warmup_steps warmup_ratio lr_scheduler_type
      eval_strategy eval_steps save_strategy save_steps save_total_limit
      logging_steps logging_first_step seed device no_mps dataloader_drop_last label_names
    ]
    missing = expected_keys - Trainers::TrainingArguments::DEFAULTS.keys
    assert_empty missing, "#{MESSAGE}: DEFAULTS lost #{missing.inspect}"

    accessors = expected_keys.flat_map { |key| [key, :"#{key}="] }
    assert_public_methods Trainers::TrainingArguments, accessors + %i[resolved_device to_h]
    assert_params [[:keyrest, :kwargs]], Trainers::TrainingArguments.instance_method(:initialize)
  end

  def test_lora_model
    assert_params [[:req, :model], [:req, :config]], Trainers::LoraModel.method(:apply)
    assert_params [[:req, :model]], Trainers::LoraModel.method(:merge)
    assert_params [[:req, :model], [:req, :output_dir], [:key, :config]], Trainers::LoraModel.method(:save_adapters)
    assert_params [[:req, :model], [:req, :input_dir]], Trainers::LoraModel.method(:load_adapters)
  end

  def test_lora_config
    expected = %i[r lora_alpha lora_dropout target_modules bias task_type]
    assert_empty expected - Trainers::LoraConfig::DEFAULTS.keys, MESSAGE
    assert_public_methods Trainers::LoraConfig, expected + %i[scaling to_h]
  end

  def test_save_utils
    assert_params [[:req, :model], [:req, :tokenizer], [:req, :output_dir], [:key, :training_args]],
                  Trainers::SaveUtils.method(:save_pretrained)
    assert_params [[:req, :model], [:req, :output_dir]], Trainers::SaveUtils.method(:save_lora_adapters)
    assert_params [[:req, :model], [:req, :input_dir]], Trainers::SaveUtils.method(:load_lora_adapters)
  end

  def test_callback_hooks
    hooks = %i[on_train_begin on_train_end on_epoch_begin on_epoch_end on_step_begin
               on_step_end on_log on_evaluate on_save]
    assert_public_methods Trainers::TrainerCallback, hooks

    hooks.each do |hook|
      params = Trainers::TrainerCallback.instance_method(hook).parameters
      assert_equal [[:req, :args], [:req, :state], [:req, :control]], params.first(3), "#{MESSAGE}: #{hook}"
    end
    assert_includes Trainers::TrainerCallback.instance_method(:on_log).parameters, [:key, :logs], MESSAGE
    assert_includes Trainers::TrainerCallback.instance_method(:on_evaluate).parameters, [:key, :metrics], MESSAGE
    assert_params [[:key, :patience], [:key, :threshold], [:key, :metric_name]],
                  Trainers::EarlyStoppingCallback.instance_method(:initialize)
  end

  def test_public_constants
    %i[Trainer TrainingArguments TrainerCallback PrinterCallback LoggingCallback EarlyStoppingCallback TrainerState
       TrainerControl LoraConfig LoraModel LoraLinear Dataset DataCollatorWithPadding DefaultDataCollator
       EvalPrediction SaveUtils].each do |name|
      assert Trainers.const_defined?(name, false), "#{MESSAGE}: Trainers::#{name} is missing"
    end
  end

  private

  def assert_public_methods(klass, expected)
    missing = expected - klass.public_instance_methods
    assert_empty missing, "#{MESSAGE}: #{klass} lost #{missing.inspect}"
  end

  def assert_params(expected, method)
    assert_equal expected, method.parameters, "#{MESSAGE}: #{method.owner}##{method.name}"
  end
end
