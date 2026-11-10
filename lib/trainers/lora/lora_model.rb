# frozen_string_literal: true

module Trainers
  class LoraModel
    # Apply LoRA adapters to a model.
    #
    # 1. Finds all Linear layers matching config.target_modules
    # 2. Replaces each with a LoraLinear that freezes the original weight
    # 3. Freezes all base model parameters
    # 4. LoRA A/B matrices stay trainable, and so do the modules named by
    #    config.modules_to_save (by default the task head for task_type)
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

      unfreeze_modules_to_save(model, config)

      Trainers.log(:info, "LoRA applied to #{targets.size} modules: #{targets.keys.join(', ')}")
      LoraUtils.print_trainable_parameters(model)

      model
    end

    # Keeps the task head trainable, as PEFT does. Matches the last segment of
    # each module name, so "classifier" matches "classifier" but not "distilbert.classifier_x".
    def self.unfreeze_modules_to_save(model, config)
      names = config.resolved_modules_to_save
      return if names.empty?

      matched = model.named_modules.select { |name, _| names.include?(name.split(".").last) }
      if matched.empty? && !config.modules_to_save.nil?
        raise AdapterError, "No modules found matching modules_to_save: #{names.inspect}"
      end

      matched.each_value { |mod| mod.parameters.each { |p| p.requires_grad = true } }
      Trainers.log(:info, "Training in full: #{matched.keys.join(', ')}") unless matched.empty?
    end
    private_class_method :unfreeze_modules_to_save

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
