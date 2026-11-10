# frozen_string_literal: true

require "json"

module Trainers
  # Internal: the training state a checkpoint needs to resume, written next to
  # the weights (full fine-tune) or the adapters (LoRA run).
  #
  #   trainer_state.json      format_version, global_step, epoch position, seed,
  #                           log history, AdamW step counts
  #   optimizer.safetensors   AdamW moments for each trainable parameter, by name
  module Checkpoint
    FORMAT_VERSION  = 1
    STATE_FILE      = "trainer_state.json"
    OPTIMIZER_FILE  = "optimizer.safetensors"
    MOMENTS         = %i[exp_avg exp_avg_sq max_exp_avg_sq].freeze

    module_function

    # Writes trainer_state.json and optimizer.safetensors into dir
    def save(dir, model:, optimizer:, state:, position:, seed:, lora:)
      require "safetensors"

      names   = model.named_parameters.to_h { |name, param| [param.object_id, name] }
      moments = {}
      steps   = {}
      optimizer_state(optimizer).each do |param, param_state|
        name = names[param.object_id]
        next if name.nil? || param_state.empty?

        steps[name] = param_state[:step]
        MOMENTS.each do |key|
          moments["#{name}.#{key}"] = param_state[key].detach.cpu if param_state[key]
        end
      end
      Safetensors::Torch.save_file(moments, File.join(dir, OPTIMIZER_FILE), metadata: { "format" => "pt" })

      data = {
        format_version:        FORMAT_VERSION,
        trainers_version:      VERSION,
        lora:                  lora,
        global_step:           state.global_step,
        max_steps:             state.max_steps,
        epochs_completed:      position[:epochs_completed],
        batches_done_in_epoch: position[:batches_done_in_epoch],
        seed:                  seed,
        best_metric:           state.best_metric,
        best_model_checkpoint: state.best_model_checkpoint,
        log_history:           state.log_history,
        optimizer_steps:       steps
      }
      File.write(File.join(dir, STATE_FILE), JSON.pretty_generate(data) + "\n")
    end

    # Reads trainer_state.json, checking it is a checkpoint this version can resume
    def read(dir)
      raise CheckpointError, "Checkpoint not found: #{dir}" unless Dir.exist?(dir)

      path = File.join(dir, STATE_FILE)
      unless File.exist?(path)
        raise CheckpointError, "#{dir} has no #{STATE_FILE}. It was written by trainers-rb older than 0.3.0, " \
                               "holds weights only, and cannot resume training."
      end

      data = JSON.parse(File.read(path), symbolize_names: true)
      version = data[:format_version]
      unless version.is_a?(Integer) && version.between?(1, FORMAT_VERSION)
        raise CheckpointError, "#{path} has checkpoint format #{version.inspect}; trainers-rb #{VERSION} " \
                               "reads format 1 to #{FORMAT_VERSION}"
      end
      data
    end

    # Restores AdamW moments and step counts onto the optimizer's parameters
    def restore_optimizer(dir, data, model:, optimizer:)
      require "safetensors"

      path = File.join(dir, OPTIMIZER_FILE)
      raise CheckpointError, "No #{OPTIMIZER_FILE} in #{dir}" unless File.exist?(path)

      moments = Safetensors::Torch.load_file(path)
      params  = model.named_parameters
      target  = optimizer_state(optimizer)

      data[:optimizer_steps].each do |name, step|
        name  = name.to_s
        param = params[name]
        unless param&.requires_grad
          raise CheckpointError, "#{path} has optimizer state for #{name}, which is not a trainable parameter " \
                                 "of this model. Resume with the same model and LoRA config."
        end

        restored = { step: step }
        MOMENTS.each do |key|
          tensor = moments["#{name}.#{key}"]
          restored[key] = tensor.to(param.device) if tensor
        end
        target[param] = restored
      end
    end

    # The checkpoint-N directory with the highest N in output_dir
    def latest(output_dir)
      Dir.glob(File.join(output_dir, "checkpoint-*"))
         .select { |dir| File.basename(dir).match?(/\Acheckpoint-\d+\z/) && File.directory?(dir) }
         .max_by { |dir| dir[/checkpoint-(\d+)\z/, 1].to_i }
    end

    # torch-rb's Optimizer#state_dict raises NotImplementedYet, so read the
    # per-parameter state hash directly. Its layout is the same from 0.17 to 0.27.
    def optimizer_state(optimizer)
      optimizer.instance_variable_get(:@state) ||
        raise(CheckpointError, "#{optimizer.class} has no per-parameter state to checkpoint")
    end
  end
end
