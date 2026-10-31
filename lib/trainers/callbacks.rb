# frozen_string_literal: true

module Trainers
  class TrainerState
    attr_accessor :epoch, :global_step, :max_steps, :num_train_epochs,
                  :total_flos, :best_metric, :best_model_checkpoint,
                  :log_history

    def initialize
      @epoch                 = 0.0
      @global_step           = 0
      @max_steps             = 0
      @num_train_epochs      = 0
      @total_flos            = 0
      @best_metric           = nil
      @best_model_checkpoint = nil
      @log_history           = []
    end
  end

  class TrainerControl
    attr_accessor :should_training_stop, :should_epoch_stop,
                  :should_save, :should_evaluate, :should_log

    def initialize
      @should_training_stop = false
      @should_epoch_stop    = false
      @should_save          = false
      @should_evaluate      = false
      @should_log           = false
    end
  end

  # Base class for trainer callbacks. Override any hook you need.
  class TrainerCallback
    def on_train_begin(args, state, control, **kwargs); end
    def on_train_end(args, state, control, **kwargs); end
    def on_epoch_begin(args, state, control, **kwargs); end
    def on_epoch_end(args, state, control, **kwargs); end
    def on_step_begin(args, state, control, **kwargs); end
    def on_step_end(args, state, control, **kwargs); end
    def on_log(args, state, control, logs: nil, **kwargs); end
    def on_evaluate(args, state, control, metrics: nil, **kwargs); end
    def on_save(args, state, control, **kwargs); end
  end

  # Prints training progress to stdout. Opt in with callbacks: [PrinterCallback.new]
  class PrinterCallback < TrainerCallback
    def on_log(args, state, control, logs: nil, **kwargs)
      return unless logs
      output = logs.map { |k, v| "#{k}: #{format_value(v)}" }.join("  ")
      emit "[step #{state.global_step}] #{output}"
    end

    def on_train_begin(args, state, control, **kwargs)
      emit "Starting training: #{state.num_train_epochs} epochs, #{state.max_steps} total steps"
    end

    def on_train_end(args, state, control, **kwargs)
      emit "Training complete. Total steps: #{state.global_step}"
    end

    def on_evaluate(args, state, control, metrics: nil, **kwargs)
      return unless metrics
      output = metrics.map { |k, v| "#{k}: #{format_value(v)}" }.join("  ")
      emit "[eval step #{state.global_step}] #{output}"
    end

    private

    def emit(message)
      puts message
    end

    # Small floats such as learning rates (2e-5) would print as 0.0000
    def format_value(v)
      return v.to_s unless v.is_a?(Float)

      v.nonzero? && v.abs < 1e-3 ? format("%.3e", v) : format("%.4f", v)
    end
  end

  # Default callback: reports the same progress through Trainers.logger
  class LoggingCallback < PrinterCallback
    private

    def emit(message)
      Trainers.log(:info, message)
    end
  end

  # Early stopping callback
  class EarlyStoppingCallback < TrainerCallback
    def initialize(patience: 3, threshold: 0.0, metric_name: "eval_loss")
      @patience    = patience
      @threshold   = threshold
      @metric_name = metric_name
      @best_value  = nil
      @wait_count  = 0
    end

    def on_evaluate(args, state, control, metrics: nil, **kwargs)
      return unless metrics

      current = metrics[@metric_name] || metrics[@metric_name.to_sym]
      return unless current

      if @best_value.nil? || improved?(current, @best_value)
        @best_value = current
        @wait_count = 0
      else
        @wait_count += 1
        if @wait_count >= @patience
          Trainers.log(:info, "Early stopping triggered after #{@wait_count} evaluations without improvement")
          control.should_training_stop = true
        end
      end
    end

    private

    def improved?(current, best)
      if @metric_name.include?("loss")
        current < best - @threshold
      else
        current > best + @threshold
      end
    end
  end

  # Dispatches callback events to all registered callbacks
  class CallbackHandler
    def initialize(callbacks)
      @callbacks = callbacks
    end

    def fire(event, args, state, control, **kwargs)
      @callbacks.each do |cb|
        cb.send(event, args, state, control, **kwargs) if cb.respond_to?(event)
      end
      control
    end
  end
end
