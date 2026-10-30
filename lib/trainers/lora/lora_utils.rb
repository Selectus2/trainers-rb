# frozen_string_literal: true

module Trainers
  module LoraUtils
    # Find all Linear modules in a model matching the target pattern
    def self.find_target_modules(model, target_modules)
      targets = {}

      model.named_modules.each do |name, mod|
        next unless mod.is_a?(Torch::NN::Linear)

        if target_modules == :all_linear
          targets[name] = mod
        elsif target_modules.is_a?(Array)
          if target_modules.any? { |pattern| name.include?(pattern) }
            targets[name] = mod
          end
        end
      end

      targets
    end

    # Replace a child module on a parent by setting the instance variable.
    # torch-rb's named_children discovers modules via instance variable scan,
    # so setting the ivar is the correct replacement mechanism.
    def self.replace_module(model, target_name, new_module)
      parts = target_name.split(".")
      parent = model

      # Navigate to the parent module
      parts[0...-1].each do |part|
        if part =~ /\A\d+\z/
          # Numeric index — likely a ModuleList element
          parent = parent.instance_variable_get(:@modules)[part.to_i]
        else
          parent = parent.instance_variable_get(:"@#{part}")
        end

        raise AdapterError, "Could not find module part '#{part}' in #{target_name}" unless parent
      end

      child_name = parts.last
      if child_name =~ /\A\d+\z/
        parent.instance_variable_get(:@modules)[child_name.to_i] = new_module
      else
        parent.instance_variable_set(:"@#{child_name}", new_module)
      end
    end

    # Count total and trainable parameters
    def self.print_trainable_parameters(model)
      total    = 0
      trainable = 0

      model.parameters.each do |p|
        total += p.numel
        trainable += p.numel if p.requires_grad
      end

      pct = total > 0 ? (trainable.to_f / total * 100) : 0.0
      Trainers.log(:info, "trainable params: #{format_number(trainable)} || " \
                          "all params: #{format_number(total)} || " \
                          "trainable%: #{format('%.4f', pct)}%")

      { total: total, trainable: trainable, percentage: pct }
    end

    def self.format_number(n)
      n.to_s.reverse.gsub(/(\d{3})(?=\d)/, '\\1,').reverse
    end
  end
end
