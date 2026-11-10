# frozen_string_literal: true

module Trainers
  class Trainer
    attr_reader :model, :args, :train_dataset, :eval_dataset, :tokenizer,
                :data_collator, :optimizer, :lr_scheduler, :state, :control

    def initialize(
      model:,
      args: nil,
      train_dataset: nil,
      eval_dataset: nil,
      tokenizer: nil,
      data_collator: nil,
      compute_metrics: nil,
      callbacks: []
    )
      @model           = model
      @args            = args || TrainingArguments.new
      @train_dataset   = train_dataset
      @eval_dataset    = eval_dataset
      @tokenizer       = tokenizer
      @data_collator   = data_collator || DefaultDataCollator.new
      @compute_metrics = compute_metrics
      @state           = TrainerState.new
      @control         = TrainerControl.new

      all_callbacks = [LoggingCallback.new] + callbacks
      @callback_handler = CallbackHandler.new(all_callbacks)
    end

    def train
      device = @args.resolved_device
      apply_seed
      @model.to(device)
      @model.train

      batch_size        = @args.per_device_train_batch_size
      accum             = @args.gradient_accumulation_steps
      batches_per_epoch = num_batches(@train_dataset, batch_size)
      if batches_per_epoch.zero?
        raise ConfigurationError, "train_dataset yields no batches: #{@train_dataset.size} examples, " \
                                  "batch size #{batch_size}, dataloader_drop_last #{@args.dataloader_drop_last}"
      end

      # One step is one optimizer update. The last accumulation window of an
      # epoch may be shorter than gradient_accumulation_steps; it still steps.
      steps_per_epoch = (batches_per_epoch.to_f / accum).ceil
      total_steps     = steps_per_epoch * @args.num_train_epochs

      @state.max_steps        = total_steps
      @state.num_train_epochs = @args.num_train_epochs

      @optimizer    = create_optimizer
      @lr_scheduler = create_scheduler(total_steps)
      @optimizer.zero_grad
      start_epoch, skip_batches = resume_from_checkpoint

      @callback_handler.fire(:on_train_begin, @args, @state, @control)

      @args.num_train_epochs.times do |epoch|
        next if epoch < start_epoch

        @state.epoch = epoch + 1
        @callback_handler.fire(:on_epoch_begin, @args, @state, @control)
        @model.train

        epoch_loss    = 0.0
        epoch_batches = 0

        skip = epoch == start_epoch ? skip_batches : 0
        batches_done = skip
        each_batch(@train_dataset, batch_size, shuffle: true, skip: skip) do |batch, index|
          batches_done = index + 1
          position     = index % accum
          window_start = index - position
          window_size  = [accum, batches_per_epoch - window_start].min

          @callback_handler.fire(:on_step_begin, @args, @state, @control) if position.zero?

          batch = move_to_device(batch, device)
          loss  = compute_loss(batch)

          # Average the gradient over the micro-batches in this window
          scaled_loss = window_size > 1 ? loss / window_size : loss
          scaled_loss.backward

          epoch_loss    += loss.item
          epoch_batches += 1

          next unless position == window_size - 1

          clip_grad_norm!(@model.parameters, @args.max_grad_norm)
          @optimizer.step
          @lr_scheduler.step
          @optimizer.zero_grad
          @state.global_step += 1

          # Logging
          if should_log?
            logs = {
              loss:          epoch_loss / epoch_batches,
              learning_rate: current_lr,
              epoch:         @state.epoch
            }
            @state.log_history << logs.merge(step: @state.global_step)
            @callback_handler.fire(:on_log, @args, @state, @control, logs: logs)
          end

          # Step-based evaluation
          if @args.eval_strategy == :steps && @args.eval_steps &&
             @state.global_step % @args.eval_steps == 0
            metrics = evaluate
            @callback_handler.fire(:on_evaluate, @args, @state, @control, metrics: metrics)
          end

          # Step-based saving
          if @args.save_strategy == :steps && @args.save_steps &&
             @state.global_step % @args.save_steps == 0
            save_checkpoint(epochs_completed: epoch, batches_done_in_epoch: index + 1)
            @callback_handler.fire(:on_save, @args, @state, @control)
          end

          @callback_handler.fire(:on_step_end, @args, @state, @control)
          break if @control.should_training_stop || @control.should_epoch_stop
        end

        # Epoch-level logging, unless the last step of the epoch already logged
        unless @state.log_history.last&.fetch(:step, nil) == @state.global_step
          epoch_avg_loss = epoch_batches > 0 ? epoch_loss / epoch_batches : 0.0
          logs = { loss: epoch_avg_loss, learning_rate: current_lr, epoch: @state.epoch }
          @state.log_history << logs.merge(step: @state.global_step)
          @callback_handler.fire(:on_log, @args, @state, @control, logs: logs)
        end

        # Epoch-based evaluation
        if @args.eval_strategy == :epoch && @eval_dataset
          metrics = evaluate
          @callback_handler.fire(:on_evaluate, @args, @state, @control, metrics: metrics)
        end

        # Epoch-based saving
        # A run stopped part-way through the epoch records where it stopped,
        # counting batches skipped when it was resumed. An epoch ended early
        # with should_epoch_stop is complete: its other batches were skipped
        # on purpose, and a resumed run must skip them too.
        if @args.save_strategy == :epoch
          if batches_done == batches_per_epoch || @control.should_epoch_stop
            save_checkpoint(epochs_completed: epoch + 1, batches_done_in_epoch: 0)
          else
            save_checkpoint(epochs_completed: epoch, batches_done_in_epoch: batches_done)
          end
          @callback_handler.fire(:on_save, @args, @state, @control)
        end

        @callback_handler.fire(:on_epoch_end, @args, @state, @control)
        @control.should_epoch_stop = false
        break if @control.should_training_stop
      end

      @callback_handler.fire(:on_train_end, @args, @state, @control)
      @state
    end

    def evaluate(eval_dataset: nil)
      dataset = eval_dataset || @eval_dataset
      raise ArgumentError, "No eval_dataset provided" unless dataset

      device = @args.resolved_device
      @model.eval

      all_preds  = []
      all_labels = []
      total_loss = 0.0
      total_steps = 0

      Torch.no_grad do
        each_batch(dataset, @args.per_device_eval_batch_size) do |batch|
          batch  = move_to_device(batch, device)
          labels = pop_labels(batch).values.first

          output = forward(batch)

          if labels
            logits = output.respond_to?(:logits) ? output.logits : output
            loss = Torch::NN::F.cross_entropy(logits, labels)
            total_loss += loss.item
            all_labels << labels.detach.cpu
          end
          total_steps += 1

          logits = output.respond_to?(:logits) ? output.logits : output
          all_preds << logits.detach.cpu
        end
      end

      @model.train

      metrics = {}
      metrics[:eval_loss] = total_loss / total_steps if total_steps > 0

      if @compute_metrics && all_preds.any? && all_labels.any?
        preds  = Torch.cat(all_preds)
        labels = Torch.cat(all_labels)
        eval_pred = EvalPrediction.new(predictions: preds, label_ids: labels)
        custom_metrics = @compute_metrics.call(eval_pred)
        metrics.merge!(custom_metrics)
      end

      metrics
    end

    def predict(test_dataset)
      device = @args.resolved_device
      @model.eval

      all_preds = []
      Torch.no_grad do
        each_batch(test_dataset, @args.per_device_eval_batch_size) do |batch|
          batch  = move_to_device(batch, device)
          output = forward(batch)
          logits = output.respond_to?(:logits) ? output.logits : output
          all_preds << logits.detach.cpu
        end
      end

      Torch.cat(all_preds)
    end

    def save_model(output_dir = nil)
      output_dir ||= @args.output_dir
      SaveUtils.save_pretrained(@model, @tokenizer, output_dir, training_args: @args)
    end

    private

    def compute_loss(batch)
      label_values = pop_labels(batch)
      labels       = label_values.values.first

      # Try passing labels to the model (some models compute loss internally)
      output = begin
        forward(batch.merge(label_values))
      rescue => e
        # If the model doesn't support labels kwarg (e.g. transformers-rb Todo),
        # fall back to forward without labels + external loss
        if e.message.include?("Todo") || e.message.include?("not implemented")
          forward(batch)
        else
          raise
        end
      end

      # Restore labels to batch for downstream use
      batch.merge!(label_values)

      if output.respond_to?(:loss) && output.loss
        output.loss
      elsif labels
        logits = output.respond_to?(:logits) ? output.logits : output
        Torch::NN::F.cross_entropy(logits, labels)
      else
        raise ConfigurationError, "Model did not return a loss and no labels found in batch. " \
              "Either pass labels in your dataset or use a model that computes loss."
      end
    end

    def forward(batch)
      if batch.is_a?(Hash)
        # Filter to only keys the model accepts, using symbol keys
        @model.call(**batch)
      else
        @model.call(batch)
      end
    end

    def create_optimizer
      Optimization.create_optimizer(@model, @args)
    end

    def create_scheduler(total_steps)
      warmup_steps = if @args.warmup_steps > 0
                       @args.warmup_steps
                     elsif @args.warmup_ratio > 0
                       (total_steps * @args.warmup_ratio).to_i
                     else
                       0
                     end

      Optimization.create_scheduler(
        @args.lr_scheduler_type,
        @optimizer,
        num_warmup_steps:    warmup_steps,
        num_training_steps:  total_steps
      )
    end

    # Removes every key named in args.label_names (symbol or string) from the
    # batch and returns them as { name_symbol => tensor }, in label_names order
    def pop_labels(batch)
      @args.label_names.each_with_object({}) do |name, labels|
        value = batch.key?(name.to_sym) ? batch.delete(name.to_sym) : batch.delete(name.to_s)
        labels[name.to_sym] = value unless value.nil?
      end
    end

    def num_batches(dataset, batch_size)
      full, rest = dataset.size.divmod(batch_size)
      rest.positive? && !@args.dataloader_drop_last ? full + 1 : full
    end

    # Yields each collated batch and its index. Batches before skip are passed
    # over without collating them, which is how a resumed epoch catches up.
    def each_batch(dataset, batch_size, shuffle: false, skip: 0)
      return enum_for(:each_batch, dataset, batch_size, shuffle: shuffle, skip: skip) unless block_given?

      indices = (0...dataset.size).to_a
      indices.shuffle!(random: @rng || Random.new) if shuffle

      num_batches(dataset, batch_size).times do |i|
        next if i < skip

        batch_indices = indices[i * batch_size, batch_size]
        features = batch_indices.map { |j| dataset[j] }
        yield @data_collator.call(features), i
      end
    end

    # Seeds Torch, Kernel#rand and the shuffle RNG from args.seed (nil skips)
    def apply_seed
      seed = @args.seed
      @rng = seed.nil? ? Random.new : Random.new(seed)
      Trainers.set_seed(seed) unless seed.nil?
    end

    def move_to_device(batch, device)
      batch.each_with_object({}) do |(key, value), result|
        result[key] = if value.is_a?(Torch::Tensor)
                        value.to(device)
                      else
                        value
                      end
      end
    end

    def clip_grad_norm!(parameters, max_norm)
      params = parameters.select { |p| p.grad }
      return 0.0 if params.empty?

      total_norm_sq = 0.0
      params.each do |p|
        total_norm_sq += p.grad.data.norm(2).item ** 2
      end
      total_norm = Math.sqrt(total_norm_sq)

      clip_coef = max_norm / (total_norm + 1e-6)
      if clip_coef < 1.0
        params.each { |p| p.grad.data.mul!(clip_coef) }
      end

      total_norm
    end

    def current_lr
      @optimizer.param_groups.first[:lr]
    end

    def should_log?
      return true if @args.logging_first_step && @state.global_step == 1
      @state.global_step % @args.logging_steps == 0
    end

    # A full fine-tune checkpoint holds the model as save_model writes it; a
    # LoRA checkpoint holds only the adapters and other trainable weights.
    # Both add the optimizer and trainer state needed to resume.
    #
    # The files are written to a hidden directory that resume_from_checkpoint
    # ignores, which is renamed to checkpoint-N once complete, so a run
    # killed mid-save leaves the previous checkpoint as the latest.
    def save_checkpoint(position)
      name = "checkpoint-#{@state.global_step}"
      dir  = File.join(@args.output_dir, name)
      tmp  = File.join(@args.output_dir, ".#{name}.tmp")
      FileUtils.rm_rf(tmp)

      lora = lora_model?
      if lora
        SaveUtils.save_lora_adapters(@model, tmp)
      else
        save_model(tmp)
      end
      Checkpoint.save(tmp, model: @model, optimizer: @optimizer, state: @state,
                           position: position, seed: @args.seed, lora: lora)

      FileUtils.rm_rf(dir)
      File.rename(tmp, dir)
      cleanup_checkpoints if @args.save_total_limit
    end

    def lora_model?
      @model.named_modules.any? { |_, mod| mod.is_a?(LoraLinear) }
    end

    # Restores weights, optimizer, scheduler, step counters and data order from
    # args.resume_from_checkpoint. Returns [epochs to skip, batches to skip in
    # the first resumed epoch]; [0, 0] when not resuming.
    def resume_from_checkpoint
      setting = @args.resume_from_checkpoint
      return [0, 0] unless setting

      dir = setting == true ? Checkpoint.latest(@args.output_dir) : setting
      if dir.nil?
        raise CheckpointError, "resume_from_checkpoint is true, but #{@args.output_dir} has no checkpoint-N directory"
      end
      data = Checkpoint.read(dir)

      if data[:lora]
        unless lora_model?
          raise CheckpointError, "#{dir} is a LoRA checkpoint. Apply LoRA to the base model with the same " \
                                 "LoraConfig before resuming."
        end
        SaveUtils.load_lora_adapters(@model, dir)
      else
        raise CheckpointError, "#{dir} is a full checkpoint, but the model has LoRA adapters" if lora_model?

        SaveUtils.load_weights(@model, File.join(dir, SaveUtils::WEIGHTS_FILE))
      end
      Checkpoint.restore_optimizer(dir, data, model: @model, optimizer: @optimizer)

      @state.global_step           = data[:global_step]
      @state.log_history           = data[:log_history]
      @state.best_metric           = data[:best_metric]
      @state.best_model_checkpoint = data[:best_model_checkpoint]
      @lr_scheduler.step(@state.global_step)

      if data[:max_steps] != @state.max_steps
        Trainers.log(:warn, "#{dir} was written by a run of #{data[:max_steps]} steps; this run has " \
                            "#{@state.max_steps}, so the learning-rate schedule differs from the original")
      end
      if @args.seed.nil? || @args.seed != data[:seed]
        Trainers.log(:warn, "seed is #{@args.seed.inspect}, but #{dir} was written with seed #{data[:seed].inspect}; " \
                            "the resumed data order differs from the original run")
      end

      # Replay the shuffles of the epochs already done, so the shuffle RNG is
      # where the original run left it. Torch's RNG cannot be saved by
      # torch-rb, so it is reseeded from the seed and step instead.
      data[:epochs_completed].times { (0...@train_dataset.size).to_a.shuffle!(random: @rng) }
      Torch.manual_seed(@args.seed + @state.global_step) unless @args.seed.nil?

      Trainers.log(:info, "Resumed from #{dir} at step #{@state.global_step} " \
                          "(epoch #{data[:epochs_completed] + 1}, batch #{data[:batches_done_in_epoch]})")
      [data[:epochs_completed], data[:batches_done_in_epoch]]
    end

    def cleanup_checkpoints
      return unless @args.save_total_limit

      checkpoints = Dir.glob(File.join(@args.output_dir, "checkpoint-*"))
                       .sort_by { |d| d[/checkpoint-(\d+)/, 1].to_i }

      while checkpoints.length > @args.save_total_limit
        old = checkpoints.shift
        FileUtils.rm_rf(old)
      end
    end
  end
end
