require_relative "test_helper"

# Records every hook with the arguments it received and a snapshot of state
class RecordingCallback < Trainers::TrainerCallback
  Event = Struct.new(:hook, :args, :state, :control, :kwargs, :global_step, :epoch, keyword_init: true)

  attr_reader :events

  def initialize
    super
    @events = []
  end

  %i[on_train_begin on_train_end on_epoch_begin on_epoch_end on_step_begin
     on_step_end on_log on_evaluate on_save].each do |hook|
    define_method(hook) do |args, state, control, **kwargs|
      @events << Event.new(hook: hook, args: args, state: state, control: control, kwargs: kwargs,
                           global_step: state.global_step, epoch: state.epoch)
    end
  end

  def hooks
    @events.map(&:hook)
  end
end

class StopAfterFirstEpoch < Trainers::TrainerCallback
  def on_epoch_end(args, state, control, **)
    control.should_training_stop = true
  end
end

class CallbacksTest < Minitest::Test
  def setup
    @tmpdir = Dir.mktmpdir
  end

  def teardown
    FileUtils.rm_rf(@tmpdir)
  end

  # 8 rows, batch size 4 => 2 steps per epoch, 2 epochs
  def run_recorded(*extra_callbacks, **arg_overrides)
    recorder = RecordingCallback.new
    args = training_args(num_train_epochs: 2, per_device_train_batch_size: 4,
                         eval_strategy: :epoch, save_strategy: :epoch, **arg_overrides)
    trainer = Trainers::Trainer.new(
      model: TinyClassifier.new, args: args,
      train_dataset: synthetic_dataset(8), eval_dataset: synthetic_dataset(4),
      callbacks: [recorder, *extra_callbacks]
    )
    quietly { trainer.train }
    [recorder, trainer]
  end

  def test_hook_sequence_across_two_epochs
    recorder, = run_recorded

    epoch = %i[on_epoch_begin on_step_begin on_step_end on_step_begin on_step_end
               on_log on_evaluate on_save on_epoch_end]
    assert_equal [:on_train_begin, *epoch, *epoch, :on_train_end], recorder.hooks
  end

  def test_each_hook_receives_args_state_control
    recorder, trainer = run_recorded

    recorder.events.each do |event|
      assert_same trainer.args, event.args, event.hook
      assert_same trainer.state, event.state, event.hook
      assert_same trainer.control, event.control, event.hook
    end
  end

  def test_documented_keywords
    recorder, = run_recorded

    recorder.events.each do |event|
      case event.hook
      when :on_log
        assert_equal [:logs], event.kwargs.keys
        assert_equal %i[loss learning_rate epoch], event.kwargs[:logs].keys
      when :on_evaluate
        assert_equal [:metrics], event.kwargs.keys
        assert_kind_of Float, event.kwargs[:metrics][:eval_loss]
      else
        assert_empty event.kwargs, "#{event.hook} should receive no keywords"
      end
    end
  end

  # With gradient_accumulation_steps = 1, micro-batches and optimizer steps
  # coincide; global_step under accumulation is open in #12.
  def test_state_values_when_hooks_fire
    recorder, = run_recorded
    by_hook = recorder.events.group_by(&:hook)

    begin_event = by_hook[:on_train_begin].first
    assert_equal 0, begin_event.global_step
    assert_equal 4, begin_event.state.max_steps
    assert_equal [1, 2], by_hook[:on_epoch_begin].map(&:epoch)
    assert_equal [0, 1, 2, 3], by_hook[:on_step_begin].map(&:global_step)
    assert_equal [1, 2, 3, 4], by_hook[:on_step_end].map(&:global_step)
    assert_equal [2, 4], by_hook[:on_epoch_end].map(&:global_step)
    assert_equal [2, 4], by_hook[:on_save].map(&:global_step)
    assert_equal 4, by_hook[:on_train_end].first.global_step
  end

  def test_should_training_stop_ends_the_run
    recorder, trainer = run_recorded(StopAfterFirstEpoch.new)

    assert_equal 1, recorder.hooks.count(:on_epoch_begin)
    assert_equal :on_train_end, recorder.hooks.last
    assert_equal 2, trainer.state.global_step
  end

  def test_early_stopping_after_patience
    callback = Trainers::EarlyStoppingCallback.new(patience: 2)
    control = Trainers::TrainerControl.new
    state = Trainers::TrainerState.new

    evaluate = ->(loss) { quietly { callback.on_evaluate(nil, state, control, metrics: { eval_loss: loss }) } }

    evaluate.(1.0)  # best
    evaluate.(0.8)  # improves, resets the count
    evaluate.(0.9)  # 1 without improvement
    refute control.should_training_stop
    evaluate.(0.85) # 2 without improvement
    assert control.should_training_stop
  end

  def test_early_stopping_respects_threshold_and_direction
    callback = Trainers::EarlyStoppingCallback.new(patience: 1, threshold: 0.05, metric_name: "accuracy")
    control = Trainers::TrainerControl.new
    state = Trainers::TrainerState.new

    quietly { callback.on_evaluate(nil, state, control, metrics: { accuracy: 0.80 }) }
    quietly { callback.on_evaluate(nil, state, control, metrics: { accuracy: 0.82 }) } # below threshold
    assert control.should_training_stop
  end

  def test_early_stopping_stops_a_real_run
    never_improves = Class.new(Trainers::TrainerCallback) do
      def on_evaluate(args, state, control, metrics: nil, **)
        metrics[:eval_loss] = 1.0 # same value every epoch
      end
    end
    recorder, = run_recorded(never_improves.new, Trainers::EarlyStoppingCallback.new(patience: 1),
                             num_train_epochs: 5)

    assert_equal 2, recorder.hooks.count(:on_epoch_begin)
  end
end
