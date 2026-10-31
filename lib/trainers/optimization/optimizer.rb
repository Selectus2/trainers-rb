# frozen_string_literal: true

module Trainers
  module Optimization
    # Builds AdamW with two param groups:
    #   - Parameters with weight decay (Linear weights, Embedding weights)
    #   - Parameters without weight decay (biases, LayerNorm/layer_norm params)
    #
    # This split is critical for transformer fine-tuning — regularizing biases
    # and normalization weights hurts convergence.
    def self.create_optimizer(model, args)
      decay_params    = []
      no_decay_params = []

      no_decay_patterns = ["bias", "LayerNorm", "layer_norm", "layernorm"]

      model.named_parameters.each do |name, param|
        next unless param.requires_grad

        if no_decay_patterns.any? { |pattern| name.include?(pattern) }
          no_decay_params << param
        else
          decay_params << param
        end
      end

      param_groups = []

      if decay_params.any?
        param_groups << { params: decay_params, weight_decay: args.weight_decay }
      end

      if no_decay_params.any?
        param_groups << { params: no_decay_params, weight_decay: 0.0 }
      end

      if param_groups.empty?
        raise ConfigurationError, "No trainable parameters found. Did you forget to unfreeze the model or apply LoRA?"
      end

      Torch::Optim::AdamW.new(
        param_groups,
        lr:    args.learning_rate,
        betas: [args.adam_beta1, args.adam_beta2],
        eps:   args.adam_epsilon
      )
    end
  end
end
