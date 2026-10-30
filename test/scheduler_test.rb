require_relative "test_helper"

class SchedulerTest < Minitest::Test
  TOTAL = 10

  # No warmup: each schedule at step 0, the midpoint and the final step
  def test_linear_curve
    assert_lrs :linear, { 0 => 1.0, 5 => 0.5, TOTAL => 0.0 }
  end

  def test_cosine_curve
    assert_lrs :cosine, { 0 => 1.0, 5 => 0.5, TOTAL => 0.0 }
  end

  def test_constant_curve
    assert_lrs :constant, { 0 => 1.0, 5 => 1.0, TOTAL => 1.0 }
  end

  # Warmup of 4 steps: step 0, warmup_steps - 1, the boundary, one past it
  def test_linear_warmup_boundaries
    assert_lrs :linear, { 0 => 0.0, 3 => 0.75, 4 => 1.0, 5 => 5.0 / 6 }, warmup: 4
  end

  def test_cosine_warmup_boundaries
    one_past = 0.5 * (1 + Math.cos(Math::PI * (1.0 / 6)))
    assert_lrs :cosine, { 0 => 0.0, 3 => 0.75, 4 => 1.0, 5 => one_past }, warmup: 4
  end

  def test_constant_warmup_boundaries
    assert_lrs :constant, { 0 => 0.0, 3 => 0.75, 4 => 1.0, 5 => 1.0 }, warmup: 4
  end

  def test_warmup_ratio_matches_equivalent_warmup_steps
    %i[linear cosine constant].each do |type|
      by_ratio = trainer_lrs(lr_scheduler_type: type, warmup_ratio: 0.4)
      by_steps = trainer_lrs(lr_scheduler_type: type, warmup_steps: 4)
      assert_equal by_steps, by_ratio, "#{type}: warmup_ratio 0.4 of #{TOTAL} steps should equal warmup_steps 4"
    end
  end

  def test_unknown_scheduler_raises_naming_valid_options
    error = assert_raises(ArgumentError) do
      Trainers::Optimization.create_scheduler(:polynomial, optimizer, num_warmup_steps: 0, num_training_steps: TOTAL)
    end
    assert_match "polynomial", error.message
    %w[:linear :cosine :constant].each { |name| assert_includes error.message, name }
  end

  private

  def optimizer
    Torch::Optim::SGD.new([Torch::NN::Parameter.new(Torch.zeros(1))], lr: 1.0)
  end

  # Learning rate (base 1.0) after each of TOTAL scheduler steps, index = step
  def lr_curve(scheduler, opt)
    lrs = [opt.param_groups.first[:lr]]
    TOTAL.times do
      opt.step
      scheduler.step
      lrs << opt.param_groups.first[:lr]
    end
    lrs
  end

  def assert_lrs(type, expected, warmup: 0)
    opt = optimizer
    scheduler = Trainers::Optimization.create_scheduler(type, opt, num_warmup_steps: warmup, num_training_steps: TOTAL)
    lrs = lr_curve(scheduler, opt)
    expected.each do |step, lr|
      assert_in_delta lr, lrs[step], 1e-9, "#{type} (warmup #{warmup}) at step #{step}"
    end
  end

  # Drives Trainer's own warmup resolution without running a training loop
  def trainer_lrs(**args)
    trainer = Trainers::Trainer.new(model: TinyClassifier.new, args: training_args(**args))
    opt = optimizer
    trainer.instance_variable_set(:@optimizer, opt)
    lr_curve(trainer.send(:create_scheduler, TOTAL), opt)
  end
end
