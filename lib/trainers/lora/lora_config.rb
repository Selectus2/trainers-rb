# frozen_string_literal: true

module Trainers
  class LoraConfig
    attr_accessor :r, :lora_alpha, :lora_dropout, :target_modules,
                  :bias, :task_type, :modules_to_save

    DEFAULTS = {
      r:              8,
      lora_alpha:     16,
      lora_dropout:   0.0,
      target_modules: ["query", "value"],  # or :all_linear
      bias:           :none,               # :none, :all, :lora_only
      task_type:      :sequence_classification,
      modules_to_save: nil                  # nil: the task head for task_type
    }.freeze

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
