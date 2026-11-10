# frozen_string_literal: true

module Trainers
  class LoraConfig
    attr_accessor :r, :lora_alpha, :lora_dropout, :target_modules,
                  :bias, :task_type, :modules_to_save

    DEFAULTS = {
      r:              8,
      lora_alpha:     16,
      lora_dropout:   0.0,
      target_modules: nil,                 # nil: DEFAULT_TARGETS for the model; or names, or :all_linear
      bias:           :none,               # :none, :all, :lora_only
      task_type:      :sequence_classification,
      modules_to_save: nil                  # nil: the task head for task_type
    }.freeze

    # Attention query and value projections for each transformers-rb
    # model_type, used when target_modules is nil
    DEFAULT_TARGETS = {
      "bert"        => %w[query value],
      "xlm-roberta" => %w[query value],
      "distilbert"  => %w[q_lin v_lin],
      "deberta-v2"  => %w[query_proj value_proj],
      "mpnet"       => %w[q v]
    }.freeze

    # Targets for models without a transformers config, the default before 0.4.0
    FALLBACK_TARGETS = %w[query value].freeze

    # Modules that stay fully trainable for each task_type, so the task head
    # (randomly initialised by from_pretrained) learns along with the adapters
    TASK_HEADS = {
      sequence_classification: %w[pre_classifier classifier score],
      token_classification:    %w[classifier],
      question_answering:      %w[qa_outputs]
    }.freeze

    def initialize(**kwargs)
      DEFAULTS.each do |key, default|
        value = kwargs.fetch(key, default)
        instance_variable_set(:"@#{key}", value)
      end
    end

    # Names of the modules to keep trainable: modules_to_save when given,
    # otherwise the task head for task_type
    def resolved_modules_to_save
      Array(@modules_to_save.nil? ? TASK_HEADS.fetch(@task_type&.to_sym, []) : @modules_to_save).map(&:to_s)
    end

    # target_modules when given, otherwise the defaults for the model's
    # config.model_type, or FALLBACK_TARGETS for a model without a config.
    # Returns [targets, source], source being :given, :model_type or :fallback.
    def resolved_target_modules(model)
      return [@target_modules, :given] unless @target_modules.nil?

      config = model.respond_to?(:config) ? model.config : nil
      return [FALLBACK_TARGETS, :fallback] unless config.respond_to?(:model_type)

      type = config.model_type.to_s
      targets = DEFAULT_TARGETS[type]
      unless targets
        raise AdapterError, "No default target_modules for model_type #{type.inspect}. Pass target_modules: with " \
                            "the Linear modules to adapt, from: #{LoraUtils.linear_names(model).first(20).join(', ')}"
      end
      [targets, :model_type]
    end

    def scaling
      @lora_alpha.to_f / @r
    end

    def to_h
      DEFAULTS.keys.each_with_object({}) do |key, hash|
        hash[key] = send(key)
      end
    end
  end
end
