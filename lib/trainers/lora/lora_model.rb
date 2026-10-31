# frozen_string_literal: true

module Trainers
  class LoraModel
    # Apply LoRA adapters to a model.
    #
    # 1. Finds all Linear layers matching config.target_modules
    # 2. Replaces each with a LoraLinear that freezes the original weight
    # 3. Freezes all base model parameters
    # 4. Only LoRA A/B matrices remain trainable
    #
    # Returns the modified model (in-place).
    def self.apply(model, config)
      config = LoraConfig.new(**config) if config.is_a?(Hash)

      # Freeze all base parameters first
      model.parameters.each { |p| p.requires_grad = false }

      # Find and replace target modules
      targets = LoraUtils.find_target_modules(model, config.target_modules)

      if targets.empty?
        raise AdapterError, "No Linear modules found matching target_modules: #{config.target_modules.inspect}. " \
              "Available modules: #{model.named_modules.map(&:first).join(', ')}"
      end

      targets.each do |name, linear|
        lora_linear = LoraLinear.new(
          linear,
          r:            config.r,
          lora_alpha:   config.lora_alpha,
          lora_dropout: config.lora_dropout
        )

        LoraUtils.replace_module(model, name, lora_linear)
      end

      # Handle bias training based on config
      case config.bias
      when :all
        model.named_parameters.each do |name, param|
          param.requires_grad = true if name.include?("bias")
        end
      when :lora_only
        model.named_modules.each do |_, mod|
          if mod.is_a?(LoraLinear) && mod.instance_variable_get(:@bias)
            mod.instance_variable_get(:@bias).requires_grad = true
          end
        end
      end
      # :none — biases stay frozen (default)

      Trainers.log(:info, "LoRA applied to #{targets.size} modules: #{targets.keys.join(', ')}")
      LoraUtils.print_trainable_parameters(model)

      model
    end

    # Merge all LoRA weights back into base weights (for inference).
    # Each LoraLinear is replaced by a plain Linear with the merged weights.
    def self.merge(model)
      lora_modules = model.named_modules.select { |_, mod| mod.is_a?(LoraLinear) }
      lora_modules.each do |name, mod|
        LoraUtils.replace_module(model, name, mod.to_linear)
      end
      Trainers.log(:info, "Merged #{lora_modules.size} LoRA adapters into base model")
      model
    end

    # Save only the LoRA adapter weights
    def self.save_adapters(model, output_dir, config: nil)
      SaveUtils.save_lora_adapters(model, output_dir)

      if config
        config_path = File.join(output_dir, "lora_config.json")
        File.write(config_path, JSON.pretty_generate(config.to_h))
      end
    end

    # Load LoRA adapter weights into a model that already has LoRA applied
    def self.load_adapters(model, input_dir)
      SaveUtils.load_lora_adapters(model, input_dir)
    end
  end
end
