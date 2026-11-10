require_relative "test_helper"

# A LoRA target with a task head named like HuggingFace's, so the default
# modules_to_save for :sequence_classification picks it up
class TinyHeadClassifier < Torch::NN::Module
  def initialize
    super()
    @encoder    = Torch::NN::Linear.new(4, 8)
    @classifier = Torch::NN::Linear.new(8, 2)
  end

  def forward(input:, labels: nil)
    @classifier.call(@encoder.call(input).relu)
  end
end

class StopAtStep < Trainers::TrainerCallback
  def initialize(step)
    super()
    @step = step
  end

  def on_step_end(args, state, control, **)
    control.should_training_stop = true if state.global_step >= @step
  end
end

# Issues #18 and #19: checkpoints resume training, and LoRA checkpoints hold adapters only
class CheckpointTest < Minitest::Test
  include TestHelpers

  # 12 rows, batch 2, accumulation 2: 6 batches and 3 optimizer steps per epoch, 9 steps in all
  RUN = {
    num_train_epochs: 3, per_device_train_batch_size: 2, gradient_accumulation_steps: 2,
    learning_rate: 0.05, weight_decay: 0.01, warmup_steps: 2, logging_steps: 1
  }.freeze

  def setup
    @tmpdir = Dir.mktmpdir
  end

  def teardown
    FileUtils.rm_rf(@tmpdir)
  end

  def test_resume_mid_epoch_matches_an_uninterrupted_run
    assert_resume_matches(save_strategy: :steps, save_steps: 4, stop_at: 4)
  end

  def test_resume_at_an_epoch_boundary_matches_an_uninterrupted_run
    assert_resume_matches(save_strategy: :epoch, stop_at: 3)
  end

  def test_resume_lora_run_matches_an_uninterrupted_run
    assert_resume_matches(save_strategy: :steps, save_steps: 4, stop_at: 4, lora: true)
  end

  # An epoch checkpoint after a mid-epoch stop records the batches done
  def test_resume_after_a_mid_epoch_stop_with_epoch_saves_matches_an_uninterrupted_run
    assert_resume_matches(save_strategy: :epoch, stop_at: 4)
    state = JSON.parse(File.read(File.join(@tmpdir, "checkpoint-4", "trainer_state.json")))
    assert_equal [1, 2], [state["epochs_completed"], state["batches_done_in_epoch"]]
  end

  # Batches skipped on resume count towards the position of the next stop
  def test_second_mid_epoch_stop_counts_the_batches_skipped_on_resume
    Trainers.set_seed(1)
    reference = train_run(TinyClassifier.new, output_dir: File.join(@tmpdir, "reference"))

    Trainers.set_seed(1)
    train_run(TinyClassifier.new, save_strategy: :steps, save_steps: 4, stop_at: 4)   # stops 2 batches into epoch 2
    train_run(TinyClassifier.new, save_strategy: :epoch, stop_at: 5, resume_from_checkpoint: true)
    state = JSON.parse(File.read(File.join(@tmpdir, "checkpoint-5", "trainer_state.json")))
    assert_equal [1, 4], [state["epochs_completed"], state["batches_done_in_epoch"]]

    resumed = train_run(TinyClassifier.new, resume_from_checkpoint: true)
    reference.model.named_parameters.each do |name, param|
      assert_equal param.to_a, resumed.model.named_parameters[name].to_a, name
    end
  end

  # A run killed while writing a checkpoint resumes from the previous one
  def test_interrupted_save_leaves_the_previous_checkpoint_as_the_latest
    original = Trainers::Checkpoint.method(:save)
    calls = 0
    Trainers::Checkpoint.define_singleton_method(:save) do |*args, **kwargs|
      calls += 1
      raise Interrupt, "killed mid-save" if calls == 2

      original.call(*args, **kwargs)
    end
    begin
      assert_raises(Interrupt) { train_run(TinyClassifier.new, save_strategy: :steps, save_steps: 2) }
    ensure
      Trainers::Checkpoint.define_singleton_method(:save, original)
    end

    assert_equal %w[.checkpoint-4.tmp checkpoint-2], Dir.children(@tmpdir).sort
    trainer = train_run(TinyClassifier.new, save_strategy: :steps, save_steps: 2, resume_from_checkpoint: true)
    assert_equal 9, trainer.state.global_step
    refute Dir.exist?(File.join(@tmpdir, ".checkpoint-4.tmp")), "the next save of step 4 replaces the partial one"
    assert Dir.exist?(File.join(@tmpdir, "checkpoint-4"))
  end

  def test_full_checkpoint_contents
    train_run(TinyClassifier.new, save_strategy: :steps, save_steps: 4, stop_at: 4)
    dir = File.join(@tmpdir, "checkpoint-4")

    assert_equal %w[model.safetensors optimizer.safetensors trainer_state.json training_args.json], Dir.children(dir).sort
    state = JSON.parse(File.read(File.join(dir, "trainer_state.json")))
    assert_equal 1, state["format_version"]
    assert_equal 4, state["global_step"]
    assert_equal 1, state["epochs_completed"]
    assert_equal 2, state["batches_done_in_epoch"]
    assert_equal 42, state["seed"]
    refute state["lora"]
    assert_equal({ "fc.weight" => 4, "fc.bias" => 4 }, state["optimizer_steps"])
  end

  def test_lora_checkpoint_holds_adapters_and_head_but_not_the_frozen_base
    model = lora_model
    train_run(model, save_strategy: :steps, save_steps: 3, stop_at: 3)
    dir = File.join(@tmpdir, "checkpoint-3")

    assert_equal %w[adapter_model.safetensors optimizer.safetensors trainer_state.json], Dir.children(dir).sort
    adapters = Safetensors::Torch.load_file(File.join(dir, "adapter_model.safetensors"))
    assert_equal %w[classifier.bias classifier.weight encoder.lora_A encoder.lora_B], adapters.keys.sort

    optimizer = Safetensors::Torch.load_file(File.join(dir, "optimizer.safetensors"))
    refute(optimizer.keys.any? { |key| key.start_with?("encoder.weight", "encoder.bias") })
  end

  def test_resume_true_picks_the_latest_checkpoint
    train_run(TinyClassifier.new, save_strategy: :steps, save_steps: 2, stop_at: 4)
    assert Dir.exist?(File.join(@tmpdir, "checkpoint-2"))

    trainer = train_run(TinyClassifier.new, save_strategy: :no, resume_from_checkpoint: true)
    assert_equal 9, trainer.state.global_step
    assert_equal [1, 2, 3, 4, 5, 6, 7, 8, 9], trainer.state.log_history.map { |log| log[:step] }.uniq
  end

  def test_resume_true_without_checkpoints_raises
    error = assert_raises(Trainers::CheckpointError) { train_run(TinyClassifier.new, resume_from_checkpoint: true) }
    assert_match "no checkpoint-N directory", error.message
  end

  def test_checkpoint_from_before_0_3_0_raises
    dir = File.join(@tmpdir, "checkpoint-1")
    quietly { Trainers::SaveUtils.save_pretrained(TinyClassifier.new, nil, dir) }

    error = assert_raises(Trainers::CheckpointError) { train_run(TinyClassifier.new, resume_from_checkpoint: dir) }
    assert_match "older than 0.3.0", error.message
  end

  def test_unknown_checkpoint_format_raises
    train_run(TinyClassifier.new, save_strategy: :steps, save_steps: 4, stop_at: 4)
    path = File.join(@tmpdir, "checkpoint-4", "trainer_state.json")
    File.write(path, JSON.generate(JSON.parse(File.read(path)).merge("format_version" => 99)))

    error = assert_raises(Trainers::CheckpointError) { train_run(TinyClassifier.new, resume_from_checkpoint: true) }
    assert_match "format 99", error.message
  end

  def test_lora_checkpoint_into_a_model_without_lora_raises
    train_run(lora_model, save_strategy: :steps, save_steps: 4, stop_at: 4)

    error = assert_raises(Trainers::CheckpointError) { train_run(TinyHeadClassifier.new, resume_from_checkpoint: true) }
    assert_match "Apply LoRA", error.message
  end

  def test_resume_argument_is_validated
    error = assert_raises(ArgumentError) { training_args(resume_from_checkpoint: 5) }
    assert_match "resume_from_checkpoint", error.message
  end

  def test_lora_keeps_the_task_head_trainable
    model = lora_model
    trainable = model.named_parameters.select { |_, param| param.requires_grad }.keys.sort
    assert_equal %w[classifier.bias classifier.weight encoder.lora_A encoder.lora_B], trainable
  end

  def test_modules_to_save_overrides_the_task_head
    model = lora_model(modules_to_save: [])
    trainable = model.named_parameters.select { |_, param| param.requires_grad }.keys.sort
    assert_equal %w[encoder.lora_A encoder.lora_B], trainable

    error = assert_raises(Trainers::AdapterError) { lora_model(modules_to_save: ["missing"]) }
    assert_match "modules_to_save", error.message
  end

  def test_saved_adapters_reproduce_the_trained_model
    model = lora_model
    train_run(model)
    quietly { Trainers::LoraModel.save_adapters(model, File.join(@tmpdir, "adapters")) }

    reloaded = lora_model
    quietly { Trainers::LoraModel.load_adapters(reloaded, File.join(@tmpdir, "adapters")) }
    input = Torch.randn(3, 4)
    model.eval
    reloaded.eval
    assert_equal model.(input: input).to_a, reloaded.(input: input).to_a
  end

  private

  # The resumed model starts from different weights, so any state the
  # checkpoint fails to restore shows up as a mismatch. Neither model has
  # dropout, so the runs must agree exactly.
  def assert_resume_matches(stop_at:, lora: false, **save)
    build = -> { lora ? lora_model : TinyClassifier.new }

    Trainers.set_seed(1)
    reference = train_run(build.call, output_dir: File.join(@tmpdir, "reference"))

    Trainers.set_seed(1)
    train_run(build.call, **save, stop_at: stop_at)

    Trainers.set_seed(2) unless lora # LoRA resumes onto the same frozen base
    resumed = train_run(build.call, resume_from_checkpoint: true)

    assert_equal 9, resumed.state.global_step
    assert_equal lrs(reference), lrs(resumed)
    reference.model.named_parameters.each do |name, param|
      assert_equal param.to_a, resumed.model.named_parameters[name].to_a, name
    end
  end

  def train_run(model, stop_at: nil, **overrides)
    args = training_args(**RUN, **overrides.merge(output_dir: overrides.fetch(:output_dir, @tmpdir)))
    callbacks = stop_at ? [StopAtStep.new(stop_at)] : []
    trainer = Trainers::Trainer.new(model: model, args: args, train_dataset: synthetic_dataset(12), callbacks: callbacks)
    quietly { trainer.train }
    trainer
  end

  def lora_model(**config)
    Trainers.set_seed(1)
    model = TinyHeadClassifier.new
    quietly { Trainers::LoraModel.apply(model, Trainers::LoraConfig.new(r: 2, target_modules: ["encoder"], **config)) }
    model
  end

  # Learning rate at each optimizer step
  def lrs(trainer)
    trainer.state.log_history.to_h { |log| [log[:step], log[:learning_rate]] }
  end
end
