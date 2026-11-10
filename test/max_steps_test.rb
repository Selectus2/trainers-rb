require_relative "test_helper"

# Issue #34: max_steps overrides num_train_epochs
class MaxStepsTest < Minitest::Test
  # 12 rows, batch 2, accumulation 2: 3 optimizer steps per epoch
  RUN = { per_device_train_batch_size: 2, gradient_accumulation_steps: 2, learning_rate: 0.05, logging_steps: 1 }.freeze

  def setup
    @tmpdir = Dir.mktmpdir
  end

  def teardown
    FileUtils.rm_rf(@tmpdir)
  end

  def test_shorter_than_one_epoch
    trainer = train_run(max_steps: 2)
    assert_equal 2, trainer.state.global_step
    assert_equal 1, trainer.state.num_train_epochs
  end

  def test_longer_than_one_epoch_stops_mid_epoch
    trainer = train_run(max_steps: 7, num_train_epochs: 1)
    assert_equal 7, trainer.state.global_step
    assert_equal 3, trainer.state.num_train_epochs
    assert_equal [1, 2, 3, 4, 5, 6, 7], trainer.state.log_history.map { |log| log[:step] }.uniq
  end

  def test_linear_schedule_reaches_zero_at_max_steps
    trainer = train_run(max_steps: 5, warmup_ratio: 0.2)
    lrs = trainer.state.log_history.to_h { |log| [log[:step], log[:learning_rate]] }
    assert_in_delta 0.05, lrs[1], 1e-9 # one warmup step
    assert_in_delta 0.0, lrs[5], 1e-9
  end

  def test_resume_matches_an_uninterrupted_run
    Trainers.set_seed(1)
    reference = train_run(max_steps: 7, output_dir: File.join(@tmpdir, "reference"))

    Trainers.set_seed(1)
    train_run(max_steps: 7, save_strategy: :steps, save_steps: 4, callbacks: [StopAt.new(4)])

    Trainers.set_seed(2)
    resumed = train_run(max_steps: 7, resume_from_checkpoint: true)

    assert_equal 7, resumed.state.global_step
    reference.model.named_parameters.each do |name, param|
      assert_equal param.to_a, resumed.model.named_parameters[name].to_a, name
    end
  end

  def test_resuming_a_finished_run_takes_no_more_steps
    train_run(max_steps: 4, save_strategy: :steps, save_steps: 4)
    assert_equal 4, train_run(max_steps: 4, resume_from_checkpoint: true).state.global_step
  end

  # An epoch-end checkpoint after a mid-epoch stop records where it stopped
  def test_epoch_checkpoint_after_a_mid_epoch_stop_records_the_position
    train_run(max_steps: 4, save_strategy: :epoch)
    state = JSON.parse(File.read(File.join(@tmpdir, "checkpoint-4", "trainer_state.json")))
    assert_equal [1, 2], [state["epochs_completed"], state["batches_done_in_epoch"]]
  end

  def test_validation
    [0, -1, 2.5, "3"].each do |value|
      error = assert_raises(ArgumentError) { training_args(max_steps: value) }
      assert_match "max_steps must be nil or a positive Integer", error.message
    end
  end

  private

  class StopAt < Trainers::TrainerCallback
    def initialize(step)
      super()
      @step = step
    end

    def on_step_end(args, state, control, **)
      control.should_training_stop = true if state.global_step >= @step
    end
  end

  def train_run(callbacks: [], **overrides)
    args = training_args(**RUN, **overrides.merge(output_dir: overrides.fetch(:output_dir, @tmpdir)))
    trainer = Trainers::Trainer.new(model: TinyClassifier.new, args: args, train_dataset: synthetic_dataset(12), callbacks: callbacks)
    quietly { trainer.train }
    trainer
  end
end
