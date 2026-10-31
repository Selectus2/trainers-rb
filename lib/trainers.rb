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

  # Convenience method: load model + tokenizer and prepare for training
  def self.from_pretrained(model_name, task: :sequence_classification, num_labels: 2)
    require "transformers-rb"

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

    [model, tokenizer]
  end
end
