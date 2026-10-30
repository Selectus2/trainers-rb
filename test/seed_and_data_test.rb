require_relative "test_helper"

class SeedTest < Minitest::Test
  def setup
    @tmpdir = Dir.mktmpdir
  end

  def teardown
    FileUtils.rm_rf(@tmpdir)
  end

  # Same initial weights, so only args.seed decides the data order
  def loss_curve(seed)
    Torch.manual_seed(0) # not set_seed: leave Ruby's RNG to the Trainer
    trainer = Trainers::Trainer.new(
      model: TinyClassifier.new,
      args: training_args(seed: seed, num_train_epochs: 2, per_device_train_batch_size: 2, logging_steps: 1),
      train_dataset: synthetic_dataset(8)
    )
    trainer.train
    trainer.state.log_history.map { |entry| entry[:loss] }
  end

  def test_same_seed_gives_identical_loss_curves
    assert_equal loss_curve(42), loss_curve(42)
  end

  def test_different_seeds_give_different_loss_curves
    refute_equal loss_curve(42), loss_curve(7)
  end

  def test_set_seed_makes_model_initialisation_reproducible
    Trainers.set_seed(5)
    a = TinyClassifier.new.parameters.map(&:detach)
    Trainers.set_seed(5)
    b = TinyClassifier.new.parameters.map(&:detach)

    a.zip(b).each { |x, y| assert Torch.equal(x, y) }
  end
end

class DataArgumentsTest < Minitest::Test
  def setup
    @tmpdir = Dir.mktmpdir
  end

  def teardown
    FileUtils.rm_rf(@tmpdir)
  end

  def global_steps(rows:, **overrides)
    args = training_args(num_train_epochs: 1, per_device_train_batch_size: 4, **overrides)
    trainer = Trainers::Trainer.new(model: TinyClassifier.new, args: args, train_dataset: synthetic_dataset(rows))
    trainer.train.global_step
  end

  def test_drop_last_skips_the_partial_batch
    assert_equal 3, global_steps(rows: 10, dataloader_drop_last: false)
    assert_equal 2, global_steps(rows: 10, dataloader_drop_last: true)
  end

  def test_drop_last_with_too_few_rows_raises
    assert_raises(Trainers::ConfigurationError) { global_steps(rows: 3, dataloader_drop_last: true) }
  end

  def test_drop_last_applies_to_evaluation
    args = training_args(per_device_eval_batch_size: 4, dataloader_drop_last: true)
    trainer = Trainers::Trainer.new(model: TinyClassifier.new, args: args)

    assert_equal 8, trainer.predict(synthetic_dataset(10)).size(0)
  end

  def test_label_names_selects_the_label_key
    rows = synthetic_dataset(8).instance_variable_get(:@data).map do |row|
      { input: row[:input], target: row[:labels] }
    end
    dataset = Trainers::Dataset.new(rows)
    args = training_args(num_train_epochs: 1, per_device_train_batch_size: 4, label_names: ["target"])
    trainer = Trainers::Trainer.new(model: TargetClassifier.new, args: args, train_dataset: dataset,
                                    eval_dataset: dataset)

    assert_equal 2, trainer.train.global_step
    assert_kind_of Float, trainer.evaluate[:eval_loss]
  end

  def test_default_label_names_ignore_other_keys
    rows = Array.new(4) { |i| { input: [0.1, 0.2, 0.3, i * 0.1], target: i % 2 } }
    trainer = Trainers::Trainer.new(model: TargetClassifier.new, args: training_args(num_train_epochs: 1),
                                    train_dataset: Trainers::Dataset.new(rows))

    # :target is not a label by default, so it reaches the model as an input
    # and no loss can be computed
    assert_raises(Trainers::ConfigurationError) { trainer.train }
  end
end
