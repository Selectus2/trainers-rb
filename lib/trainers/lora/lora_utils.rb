# frozen_string_literal: true

module Trainers
  module LoraUtils
    # Find the Linear modules matching target_modules, as PEFT does: a name
    # matches a target when it is the target or ends with ".target". A target
    # that matches nothing that way falls back to matching any name that
    # contains it, with a deprecation warning; substring matching ends in 0.5.0.
    # fallback: false turns the fallback off (for the per-architecture defaults).
    def self.find_target_modules(model, target_modules, fallback: true)
      linears = model.named_modules.select { |_, mod| mod.is_a?(Torch::NN::Linear) }
      return linears.to_h if target_modules == :all_linear

      targets = {}
      Array(target_modules).map(&:to_s).each do |pattern|
        matched = linears.select { |name, _| name == pattern || name.end_with?(".#{pattern}") }
        if matched.empty? && fallback
          matched = linears.select { |name, _| name.include?(pattern) }
          unless matched.empty?
            Trainers.log(:warn, "target_modules #{pattern.inspect} matched #{matched.map(&:first).join(', ')} by substring. " \
                                "Substring matching is deprecated and ends in 0.5.0; name the modules by their last part " \
                                "or full name instead.")
          end
        end
        matched.each { |name, mod| targets[name] = mod }
      end

      # Keep the model's module order
      linears.select { |name, _| targets.key?(name) }.to_h
    end

    # Names of the Linear modules, for error messages
    def self.linear_names(model)
      model.named_modules.select { |_, mod| mod.is_a?(Torch::NN::Linear) }.map(&:first)
    end

    # Replace a child module on a parent by setting the instance variable.
    # torch-rb's named_children discovers modules via instance variable scan,
    # so setting the ivar is the correct replacement mechanism.
    def self.replace_module(model, target_name, new_module)
      parts = target_name.split(".")
      parent = model

      # Navigate to the parent module
      parts[0...-1].each do |part|
        parent = child_module(parent, part)
        raise AdapterError, "Could not find module part '#{part}' in #{target_name}" unless parent
      end

      child_name = parts.last
      registered = parent.instance_variable_get(:@modules)
      if registered&.key?(child_name)
        registered[child_name] = new_module
      else
        parent.instance_variable_set(:"@#{child_name}", new_module)
      end
    end

    # ModuleList and add_module children live in @modules, a Hash keyed by
    # name ("0", "1", ...); everything else is an instance variable.
    def self.child_module(parent, name)
      registered = parent.instance_variable_get(:@modules)
      return registered[name] if registered&.key?(name)

      parent.instance_variable_get(:"@#{name}") if name.match?(/\A[a-z_]\w*\z/i)
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
