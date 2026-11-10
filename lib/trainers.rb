# frozen_string_literal: true

require "torch"
require "json"
require "fileutils"

require_relative "trainers/version"
require_relative "trainers/errors"
require_relative "trainers/logging"
require_relative "trainers/trainer_utils"
require_relative "trainers/training_arguments"
require_relative "trainers/data/dataset"
require_relative "trainers/data/data_collator"
require_relative "trainers/optimization/optimizer"
require_relative "trainers/optimization/scheduler"
require_relative "trainers/callbacks"
require_relative "trainers/save_utils"
require_relative "trainers/checkpoint"
require_relative "trainers/lora/lora_config"
require_relative "trainers/lora/lora_linear"
require_relative "trainers/lora/lora_utils"
require_relative "trainers/lora/lora_model"
require_relative "trainers/trainer"

module Trainers
  # Seeds Torch and Ruby's global RNG. Call before building a model to make
  # its initial weights reproducible; Trainer#train also calls it with args.seed.
  def self.set_seed(seed)
    Torch.manual_seed(seed)
    srand(seed)
    seed
  end

  # Convenience method: load model + tokenizer and prepare for training.
  # labels: names the classes, in id order, so the saved config.json carries
  # id2label and label2id; it sets num_labels to labels.size.
  def self.from_pretrained(model_name, task: :sequence_classification, num_labels: nil, labels: nil)
    require "transformers-rb"

    labels = label_names(labels, num_labels)
    num_labels ||= labels ? labels.size : 2

    model_class = case task
                  when :sequence_classification
                    Transformers::AutoModelForSequenceClassification
                  when :token_classification
                    Transformers::AutoModelForTokenClassification
                  when :question_answering
                    Transformers::AutoModelForQuestionAnswering
                  else
                    Transformers::AutoModel
                  end

    model     = model_class.from_pretrained(model_name, num_labels: num_labels)
    tokenizer = Transformers::AutoTokenizer.from_pretrained(model_name)

    # transformers-rb 0.2.0 config has no id2label or label2id writer, and
    # from_pretrained drops them when passed as options
    if labels
      model.config.instance_variable_set(:@id2label, labels.each_with_index.to_h { |name, id| [id, name] })
      model.config.instance_variable_set(:@label2id, labels.each_with_index.to_h)
    end

    [model, tokenizer]
  end

  # Internal: labels as unique Strings, checked against num_labels
  def self.label_names(labels, num_labels)
    return if labels.nil?

    names = Array(labels).map(&:to_s)
    raise ArgumentError, "labels must be a non-empty list of unique names, got #{labels.inspect}" if names.empty? || names.uniq.size != names.size
    if num_labels && num_labels != names.size
      raise ArgumentError, "num_labels: #{num_labels} contradicts labels: #{names.inspect} (#{names.size} labels)"
    end

    names
  end
  private_class_method :label_names

  # Loads a directory written by Trainer#save_model. Returns [model, tokenizer].
  def self.load_pretrained(model_dir)
    SaveUtils.load_pretrained(model_dir)
  end
end
