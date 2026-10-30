# frozen_string_literal: true

module Trainers
  class TrainingArguments
    DEFAULTS = {
      output_dir:                    "./output",
      num_train_epochs:              3,
      per_device_train_batch_size:   8,
      per_device_eval_batch_size:    8,
      learning_rate:                 5e-5,
      weight_decay:                  0.0,
      adam_beta1:                    0.9,
      adam_beta2:                    0.999,
      adam_epsilon:                  1e-8,
      max_grad_norm:                 1.0,
      gradient_accumulation_steps:   1,
      warmup_steps:                  0,
      warmup_ratio:                  0.0,
      lr_scheduler_type:             :linear,
      eval_strategy:                 :no,
      eval_steps:                    nil,
      save_strategy:                 :epoch,
      save_steps:                    500,
      save_total_limit:              nil,
      logging_steps:                 500,
      logging_first_step:            false,
      seed:                          42,
      device:                        nil,
      no_mps:                        false,
      dataloader_drop_last:          false,
      label_names:                   ["labels"]
    }.freeze

    SCHEDULER_TYPES = %i[linear cosine constant].freeze
    STRATEGIES      = %i[no epoch steps].freeze

    POSITIVE_NUMBERS  = %i[learning_rate].freeze
    POSITIVE_INTEGERS = %i[num_train_epochs per_device_train_batch_size per_device_eval_batch_size
                           gradient_accumulation_steps logging_steps save_steps].freeze
    NON_NEGATIVE      = %i[weight_decay max_grad_norm].freeze

    DEFAULTS.each_key do |key|
      attr_accessor key
    end

    def initialize(**kwargs)
      unknown = kwargs.keys - DEFAULTS.keys
      raise ArgumentError, "Unknown arguments: #{unknown.join(', ')}" unless unknown.empty?

      DEFAULTS.each do |key, default|
        value = kwargs.fetch(key, default)
        instance_variable_set(:"@#{key}", value)
      end

      validate!
    end

    def resolved_device
      return @device if @device

      if !@no_mps && defined?(Torch::Backends::MPS) && Torch::Backends::MPS.available?
        Torch.device("mps")
      else
        Torch.device("cpu")
      end
    end

    def to_h
      DEFAULTS.keys.each_with_object({}) do |key, hash|
        hash[key] = send(key)
      end
    end

    private

    # Rejects values that cannot produce a sensible run. Errors name the
    # argument, the value received, and what was expected.
    def validate!
      POSITIVE_NUMBERS.each do |key|
        check(key, "a positive number") { |v| v.is_a?(Numeric) && v.positive? }
      end
      POSITIVE_INTEGERS.each do |key|
        check(key, "a positive Integer") { |v| v.is_a?(Integer) && v.positive? }
      end
      NON_NEGATIVE.each do |key|
        check(key, "a non-negative number") { |v| v.is_a?(Numeric) && !v.negative? }
      end
      check(:warmup_steps, "a non-negative Integer") { |v| v.is_a?(Integer) && !v.negative? }
      check(:warmup_ratio, "a number between 0 and 1") { |v| v.is_a?(Numeric) && v.between?(0, 1) }
      check(:eval_steps, "nil or a positive Integer") { |v| v.nil? || (v.is_a?(Integer) && v.positive?) }

      check(:lr_scheduler_type, "one of #{SCHEDULER_TYPES.map(&:inspect).join(', ')}") { |v| SCHEDULER_TYPES.include?(v) }
      check(:eval_strategy, "one of #{STRATEGIES.map(&:inspect).join(', ')}") { |v| STRATEGIES.include?(v) }
      check(:save_strategy, "one of #{STRATEGIES.map(&:inspect).join(', ')}") { |v| STRATEGIES.include?(v) }
    end

    def check(key, expected)
      value = send(key)
      return if yield(value)

      raise ArgumentError, "#{key} must be #{expected}, got #{value.inspect}"
    end
  end
end
