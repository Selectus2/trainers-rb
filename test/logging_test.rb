require_relative "test_helper"

class LoggingTest < Minitest::Test
  def setup
    @tmpdir = Dir.mktmpdir
    @saved_logger = Trainers.logger
  end

  def teardown
    Trainers.logger = @saved_logger
    FileUtils.rm_rf(@tmpdir)
  end

  def test_default_logger_writes_info_to_stderr
    logger = Trainers.default_logger
    assert_equal Logger::INFO, logger.level
    assert_same $stderr, logger.instance_variable_get(:@logdev).dev
  end

  # Training, LoRA and saving with the logger off write nothing anywhere
  def test_nil_logger_is_completely_silent
    Trainers.logger = nil

    out, err = capture_io do
      model = TinyLoraTarget.new(4)
      Trainers::LoraModel.apply(model, Trainers::LoraConfig.new(r: 2, target_modules: :all_linear))
      run_training(model: TinyClassifier.new, logging_steps: 1, eval_strategy: :epoch, save_strategy: :epoch)
      Trainers::LoraModel.save_adapters(model, @tmpdir)
      Trainers::LoraModel.load_adapters(model, @tmpdir)
      Trainers::LoraModel.merge(model)
    end

    assert_empty out
    assert_empty err
  end

  def test_custom_logger_receives_progress
    io = StringIO.new
    Trainers.logger = Logger.new(io)

    run_training(model: TinyClassifier.new)

    assert_match "Starting training", io.string
    assert_match "Training complete", io.string
    assert_match "Model saved to", io.string
  end

  def test_library_never_writes_to_stdout_by_default
    Trainers.logger = Trainers.default_logger.tap { |l| l.reopen(File::NULL) }

    out, = capture_io { run_training(model: TinyClassifier.new) }

    assert_empty out
  end

  def test_printer_callback_is_opt_in_stdout
    Trainers.logger = nil

    out, = capture_io { run_training(model: TinyClassifier.new, callbacks: [Trainers::PrinterCallback.new]) }

    assert_match "Starting training", out
  end

  private

  def run_training(model:, callbacks: [], **overrides)
    args = training_args(num_train_epochs: 1, per_device_train_batch_size: 4, save_strategy: :epoch, **overrides)
    Trainers::Trainer.new(model: model, args: args, train_dataset: synthetic_dataset(8),
                          eval_dataset: synthetic_dataset(4), callbacks: callbacks).train
  end
end
