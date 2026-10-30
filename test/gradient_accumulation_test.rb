require_relative "test_helper"

class StepCounter < Trainers::TrainerCallback
  attr_reader :steps, :step_begins, :logs, :grad_sums_at_epoch_end, :lrs

  def initialize(model)
    super()
    @model = model
    @steps = 0
    @step_begins = 0
    @logs = 0
    @lrs = []
    @grad_sums_at_epoch_end = []
  end

  def on_step_begin(*, **)
    @step_begins += 1
  end

  def on_step_end(args, state, control, **)
    @steps += 1
  end

  def on_log(args, state, control, logs: nil, **)
    @logs += 1
  end

  def on_epoch_end(*, **)
    @grad_sums_at_epoch_end << @model.parameters.sum { |p| p.grad ? p.grad.abs.sum.item : 0.0 }
  end
end

class GradientAccumulationTest < Minitest::Test
  def setup
    @tmpdir = Dir.mktmpdir
  end

  def teardown
    FileUtils.rm_rf(@tmpdir)
  end

  def train(rows:, batch_size:, accum:, epochs: 2, **overrides)
    model = TinyClassifier.new
    counter = StepCounter.new(model)
    args = training_args(num_train_epochs: epochs, per_device_train_batch_size: batch_size,
                         gradient_accumulation_steps: accum, **overrides)
    trainer = Trainers::Trainer.new(model: model, args: args, train_dataset: synthetic_dataset(rows),
                                    callbacks: [counter])
    trainer.train
    [trainer, counter]
  end

  # 10 batches per epoch, accumulation 4 => windows of 4, 4 and 2
  def test_optimizer_steps_per_epoch_is_ceil_of_batches_over_accum
    trainer, counter = train(rows: 10, batch_size: 1, accum: 4)

    assert_equal 6, trainer.state.global_step
    assert_equal 6, trainer.state.max_steps
    assert_equal 6, counter.steps
    assert_equal 6, counter.step_begins
  end

  def test_final_learning_rate_matches_without_accumulation
    with_accum, = train(rows: 10, batch_size: 1, accum: 4)
    without,    = train(rows: 10, batch_size: 1, accum: 1)

    final = ->(trainer) { trainer.optimizer.param_groups.first[:lr] }
    assert_in_delta 0.0, final.(without), 1e-12
    assert_in_delta final.(without), final.(with_accum), 1e-12
  end

  def test_scheduler_is_sized_in_optimizer_steps
    trainer, = train(rows: 10, batch_size: 1, accum: 4, epochs: 1, lr_scheduler_type: :linear, learning_rate: 1.0)

    # 3 optimizer steps over a 3-step linear schedule ends at 0
    assert_in_delta 0.0, trainer.optimizer.param_groups.first[:lr], 1e-12
  end

  def test_no_gradients_survive_an_epoch_boundary
    _, counter = train(rows: 10, batch_size: 1, accum: 4)

    assert_equal [0.0, 0.0], counter.grad_sums_at_epoch_end
  end

  # Accumulating two batches of 2 equals one batch of 4 over the same rows
  def test_accumulated_update_equals_large_batch_update
    params = [[2, 2], [4, 1]].map do |batch_size, accum|
      Trainers.set_seed(0)
      model = TinyClassifier.new
      args = training_args(num_train_epochs: 1, per_device_train_batch_size: batch_size,
                           gradient_accumulation_steps: accum, seed: 3)
      Trainers::Trainer.new(model: model, args: args, train_dataset: synthetic_dataset(8)).train
      model.parameters.map { |p| p.detach.clone }
    end

    params[0].zip(params[1]).each do |a, b|
      assert Torch.allclose(a, b, rtol: 1e-5, atol: 1e-6), "max difference #{(a - b).abs.max.item}"
    end
  end

  def test_logging_steps_count_optimizer_steps
    _, counter = train(rows: 8, batch_size: 1, accum: 4, epochs: 1, logging_steps: 1)

    # 2 optimizer steps + 1 epoch-end log
    assert_equal 3, counter.logs
  end

  def test_save_steps_count_optimizer_steps
    train(rows: 8, batch_size: 1, accum: 2, epochs: 1, save_strategy: :steps, save_steps: 2)

    assert_equal %w[checkpoint-2 checkpoint-4], Dir.children(@tmpdir).sort
  end
end
