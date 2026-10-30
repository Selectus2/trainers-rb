require_relative "test_helper"

class TrainingArgumentsTest < Minitest::Test
  def test_defaults_are_valid
    args = Trainers::TrainingArguments.new
    assert_equal Trainers::TrainingArguments::DEFAULTS, args.to_h
  end

  def test_unknown_argument_raises
    error = assert_raises(ArgumentError) { Trainers::TrainingArguments.new(learning_rat: 1e-4) }
    assert_match "learning_rat", error.message
  end

  def test_rejects_non_positive_learning_rate
    [0, -1e-5, "1e-5", nil].each do |value|
      assert_rejected :learning_rate, value, "a positive number"
    end
  end

  def test_rejects_non_positive_integer_counts
    %i[num_train_epochs per_device_train_batch_size per_device_eval_batch_size
       gradient_accumulation_steps logging_steps save_steps].each do |key|
      [0, -1, 1.5, "8"].each do |value|
        assert_rejected key, value, "a positive Integer"
      end
    end
  end

  def test_rejects_negative_regularisation
    %i[weight_decay max_grad_norm].each do |key|
      assert_rejected key, -0.1, "a non-negative number"
    end
  end

  def test_rejects_bad_warmup
    assert_rejected :warmup_steps, -1, "a non-negative Integer"
    assert_rejected :warmup_steps, 2.5, "a non-negative Integer"
    assert_rejected :warmup_ratio, -0.1, "a number between 0 and 1"
    assert_rejected :warmup_ratio, 1.5, "a number between 0 and 1"
  end

  def test_rejects_bad_eval_steps
    assert_rejected :eval_steps, 0, "nil or a positive Integer"
  end

  def test_rejects_unknown_scheduler_type_at_construction
    assert_rejected :lr_scheduler_type, :cosine_with_restarts, "one of :linear, :cosine, :constant"
    assert_rejected :lr_scheduler_type, "linear", "one of :linear, :cosine, :constant"
  end

  def test_rejects_unknown_strategies
    %i[eval_strategy save_strategy].each do |key|
      assert_rejected key, :epochs, "one of :no, :epoch, :steps"
    end
  end

  def test_rejects_bad_seed
    assert_rejected :seed, -1, "nil or a non-negative Integer"
    assert_rejected :seed, "42", "nil or a non-negative Integer"
    assert_nil Trainers::TrainingArguments.new(seed: nil).seed
  end

  def test_rejects_bad_label_names
    assert_rejected :label_names, "labels", "an Array of Strings or Symbols"
    assert_rejected :label_names, [1], "an Array of Strings or Symbols"
    assert_equal [:start_positions, "end_positions"],
                 Trainers::TrainingArguments.new(label_names: [:start_positions, "end_positions"]).label_names
  end

  def test_accepts_boundary_values
    args = Trainers::TrainingArguments.new(
      weight_decay: 0, max_grad_norm: 0.0, warmup_steps: 0, warmup_ratio: 1.0,
      eval_strategy: :steps, eval_steps: 1, save_strategy: :no, lr_scheduler_type: :constant
    )
    assert_equal :constant, args.lr_scheduler_type
  end

  private

  def assert_rejected(key, value, expected)
    error = assert_raises(ArgumentError, "expected #{key}: #{value.inspect} to be rejected") do
      Trainers::TrainingArguments.new(key => value)
    end
    assert_equal "#{key} must be #{expected}, got #{value.inspect}", error.message
  end
end
