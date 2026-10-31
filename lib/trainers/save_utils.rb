# frozen_string_literal: true

require "json"
require "fileutils"

module Trainers
  module SaveUtils
    def self.save_pretrained(model, tokenizer, output_dir, training_args: nil)
      FileUtils.mkdir_p(output_dir)

      # Save model weights via safetensors
      state_dict = model.state_dict
      tensors = {}
      state_dict.each do |name, param|
        tensors[name] = param.is_a?(Torch::NN::Parameter) ? param.data : param
      end

      model_path = File.join(output_dir, "model.safetensors")
      require "safetensors"
      Safetensors::Torch.save_file(tensors, model_path)
      Trainers.log(:info, "Model saved to #{model_path}")

      # Save tokenizer
      if tokenizer
        if tokenizer.respond_to?(:save)
          tokenizer.save(File.join(output_dir, "tokenizer.json"))
        end
      end

      # Save training args
      if training_args
        args_path = File.join(output_dir, "training_args.json")
        File.write(args_path, JSON.pretty_generate(training_args.to_h))
      end

      output_dir
    end

    def self.save_lora_adapters(model, output_dir)
      FileUtils.mkdir_p(output_dir)

      lora_state = {}
      model.named_modules.each do |name, mod|
        if mod.is_a?(Trainers::LoraLinear)
          mod.lora_state_dict.each do |param_name, tensor|
            lora_state["#{name}.#{param_name}"] = tensor
          end
        end
      end

      if lora_state.empty?
        raise AdapterError, "No LoRA adapters found in model. Apply them with LoraModel.apply before saving."
      end

      adapter_path = File.join(output_dir, "adapter_model.safetensors")
      require "safetensors"
      Safetensors::Torch.save_file(lora_state, adapter_path)
      Trainers.log(:info, "LoRA adapters saved to #{adapter_path} (#{lora_state.size} tensors)")

      output_dir
    end

    def self.load_lora_adapters(model, input_dir)
      adapter_path = File.join(input_dir, "adapter_model.safetensors")
      raise AdapterError, "Adapter file not found: #{adapter_path}" unless File.exist?(adapter_path)

      require "safetensors"
      lora_state = Safetensors::Torch.load_file(adapter_path)

      model.named_modules.each do |name, mod|
        if mod.is_a?(Trainers::LoraLinear)
          a_key = "#{name}.lora_A"
          b_key = "#{name}.lora_B"
          if lora_state[a_key] && lora_state[b_key]
            mod.load_lora_weights(lora_state[a_key], lora_state[b_key])
          end
        end
      end

      Trainers.log(:info, "LoRA adapters loaded from #{adapter_path}")
    end
  end
end
