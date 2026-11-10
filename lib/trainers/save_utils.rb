# frozen_string_literal: true

require "json"
require "fileutils"

module Trainers
  module SaveUtils
    CONFIG_FILE           = "config.json"
    WEIGHTS_FILE          = "model.safetensors"
    TOKENIZER_FILE        = "tokenizer.json"
    TOKENIZER_CONFIG_FILE = "tokenizer_config.json"

    # Writes a directory that load_pretrained, transformers-rb and Python
    # transformers can read: config.json, model.safetensors, tokenizer.json,
    # tokenizer_config.json and training_args.json
    def self.save_pretrained(model, tokenizer, output_dir, training_args: nil)
      FileUtils.mkdir_p(output_dir)

      # Save model weights via safetensors. The "format" metadata is required
      # by the transformers loaders in both Ruby and Python.
      state_dict = model.state_dict
      tensors = {}
      state_dict.each do |name, param|
        tensors[name] = param.is_a?(Torch::NN::Parameter) ? param.data : param
      end

      model_path = File.join(output_dir, WEIGHTS_FILE)
      require "safetensors"
      Safetensors::Torch.save_file(tensors, model_path, metadata: { "format" => "pt" })
      Trainers.log(:info, "Model saved to #{model_path}")

      save_config(model, output_dir)
      save_tokenizer(tokenizer, output_dir) if tokenizer

      # Save training args
      if training_args
        args_path = File.join(output_dir, "training_args.json")
        File.write(args_path, JSON.pretty_generate(training_args.to_h))
      end

      output_dir
    end

    # Reads a directory written by save_pretrained. Returns [model, tokenizer];
    # tokenizer is nil when none was saved. The model is in eval mode on the CPU.
    def self.load_pretrained(input_dir)
      require "transformers-rb"
      require "safetensors"

      config_path  = File.join(input_dir, CONFIG_FILE)
      weights_path = File.join(input_dir, WEIGHTS_FILE)
      unless File.exist?(config_path)
        raise CheckpointError, "No #{CONFIG_FILE} in #{input_dir}. It was saved by trainers-rb older than 0.3.0, " \
                               "or from a model without a transformers config, and cannot be loaded."
      end
      raise CheckpointError, "No #{WEIGHTS_FILE} in #{input_dir}" unless File.exist?(weights_path)

      config_hash  = JSON.parse(File.read(config_path)).transform_keys(&:to_sym)
      architecture = Array(config_hash[:architectures]).first
      model_class  = architecture && transformers_class(architecture)
      config_class = transformers_class(Transformers::CONFIG_MAPPING_NAMES[config_hash[:model_type].to_s].to_s)
      unless model_class && config_class
        raise CheckpointError, "#{config_path} names architecture #{architecture.inspect} and model_type " \
                               "#{config_hash[:model_type].inspect}, which transformers-rb #{Transformers::VERSION} does not provide"
      end

      model = model_class.new(config_class.new(**config_hash))
      load_weights(model, weights_path)
      model.eval

      [model, load_tokenizer(input_dir)]
    end

    # Internal: loads model.safetensors into an existing model. Used by
    # load_pretrained and by Trainer when resuming a full fine-tune.
    def self.load_weights(model, weights_path)
      require "safetensors"
      raise CheckpointError, "No #{File.basename(weights_path)} in #{File.dirname(weights_path)}" unless File.exist?(weights_path)

      state = Safetensors::Torch.load_file(weights_path)
      if state.keys.any? { |key| key.end_with?(".lora_A", ".lora_B") }
        raise AdapterError, "#{weights_path} holds unmerged LoRA adapters. Merge them with LoraModel.merge " \
                            "before save_model, or save and load the adapters with LoraModel.save_adapters."
      end

      # torch-rb's state_dict leaves out non-persistent buffers such as
      # position_ids, but its load_state_dict is strict and expects them.
      # They are rebuilt by the constructor, so take them from the model.
      missing_params = model.named_parameters.keys - state.keys
      unless missing_params.empty?
        raise CheckpointError, "#{weights_path} is missing weights for #{model.class}: #{missing_params.first(5).join(', ')}"
      end
      model.named_modules.each do |name, mod|
        mod.named_buffers(prefix: name.empty? ? "" : "#{name}.").each { |key, buffer| state[key] ||= buffer }
      end

      begin
        Torch.no_grad { model.load_state_dict(state) }
      rescue StandardError => e
        raise CheckpointError, "#{weights_path} does not match #{model.class}: #{e.message}"
      end
      model
    end

    # Internal: config.json from the model's transformers config, with
    # "architectures" filled in so loaders know which class to build
    def self.save_config(model, output_dir)
      config = model.respond_to?(:config) ? model.config : nil
      unless config.respond_to?(:to_json_string)
        Trainers.log(:warn, "#{model.class} has no transformers config, so #{CONFIG_FILE} was not written " \
                            "and #{output_dir} cannot be loaded with load_pretrained")
        return
      end

      data = JSON.parse(config.to_json_string)
      data["architectures"] ||= [model.class.name.split("::").last]
      File.write(File.join(output_dir, CONFIG_FILE), JSON.pretty_generate(data.sort.to_h) + "\n")
    end

    # Internal: tokenizer.json from the tokenizers backend, plus
    # tokenizer_config.json for transformers tokenizers
    def self.save_tokenizer(tokenizer, output_dir)
      backend = tokenizer.respond_to?(:backend_tokenizer) ? tokenizer.backend_tokenizer : tokenizer
      unless backend.respond_to?(:save)
        Trainers.log(:warn, "#{tokenizer.class} cannot be saved, so #{TOKENIZER_FILE} was not written")
        return
      end
      backend.save(File.join(output_dir, TOKENIZER_FILE))
      return if backend.equal?(tokenizer)

      settings = tokenizer.respond_to?(:init_kwargs) ? tokenizer.init_kwargs.dup : {}
      settings.delete(:name_or_path)
      settings[:tokenizer_class] = tokenizer.class.name.split("::").last
      File.write(File.join(output_dir, TOKENIZER_CONFIG_FILE), JSON.pretty_generate(settings.sort.to_h) + "\n")
    end

    # Internal: nil when the directory holds no tokenizer
    def self.load_tokenizer(input_dir)
      tokenizer_path = File.join(input_dir, TOKENIZER_FILE)
      return unless File.exist?(tokenizer_path)

      settings_path = File.join(input_dir, TOKENIZER_CONFIG_FILE)
      return Tokenizers.from_file(tokenizer_path) unless File.exist?(settings_path)

      settings = JSON.parse(File.read(settings_path)).transform_keys(&:to_sym)
      name = settings.delete(:tokenizer_class).to_s
      tokenizer_class = transformers_class(name)
      raise CheckpointError, "#{settings_path} names tokenizer #{name.inspect}, which transformers-rb does not provide" unless tokenizer_class

      tokenizer_class.new(tokenizer_file: tokenizer_path, **settings)
    end

    # Internal: Transformers::Distilbert::DistilBertConfig from "DistilBertConfig"
    def self.transformers_class(name)
      return if name.empty?
      return Transformers.const_get(name, false) if Transformers.const_defined?(name, false)

      Transformers.constants.each do |const|
        namespace = Transformers.const_get(const, false)
        next unless namespace.instance_of?(Module)
        return namespace.const_get(name, false) if namespace.const_defined?(name, false)
      end
      nil
    end
    private_class_method :save_config, :save_tokenizer, :load_tokenizer, :transformers_class

    ADAPTER_FILE = "adapter_model.safetensors"

    # Writes the LoRA A/B matrices and every other trainable parameter, such
    # as the task head kept by modules_to_save, so the adapters reproduce the
    # trained model on top of the base model
    def self.save_lora_adapters(model, output_dir)
      FileUtils.mkdir_p(output_dir)

      lora_state = {}
      model.named_modules.each do |name, mod|
        if mod.is_a?(Trainers::LoraLinear)
          mod.lora_state_dict.each do |param_name, tensor|
            lora_state["#{name}.#{param_name}"] = tensor.detach.cpu
          end
        end
      end

      if lora_state.empty?
        raise AdapterError, "No LoRA adapters found in model. Apply them with LoraModel.apply before saving."
      end

      model.named_parameters.each do |name, param|
        lora_state[name] ||= param.data.detach.cpu if param.requires_grad
      end

      adapter_path = File.join(output_dir, ADAPTER_FILE)
      require "safetensors"
      Safetensors::Torch.save_file(lora_state, adapter_path, metadata: { "format" => "pt" })
      Trainers.log(:info, "LoRA adapters saved to #{adapter_path} (#{lora_state.size} tensors)")

      output_dir
    end

    def self.load_lora_adapters(model, input_dir)
      adapter_path = File.join(input_dir, ADAPTER_FILE)
      raise AdapterError, "Adapter file not found: #{adapter_path}" unless File.exist?(adapter_path)

      require "safetensors"
      lora_state = Safetensors::Torch.load_file(adapter_path)
      remaining  = lora_state.dup

      model.named_modules.each do |name, mod|
        next unless mod.is_a?(Trainers::LoraLinear)

        a_key = "#{name}.lora_A"
        b_key = "#{name}.lora_B"
        if lora_state[a_key] && lora_state[b_key]
          mod.load_lora_weights(lora_state[a_key], lora_state[b_key])
          remaining.delete(a_key)
          remaining.delete(b_key)
        end
      end

      params = model.named_parameters
      unknown = remaining.keys - params.keys
      unless unknown.empty?
        raise AdapterError, "#{adapter_path} has weights the model cannot take: #{unknown.first(5).join(', ')}. " \
                            "Apply LoRA with the same LoraConfig before loading the adapters."
      end
      Torch.no_grad do
        remaining.each { |name, tensor| params[name].copy!(tensor) }
      end

      Trainers.log(:info, "LoRA adapters loaded from #{adapter_path}")
    end
  end
end
