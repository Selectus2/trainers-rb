# frozen_string_literal: true

module Trainers
  # Base class for every error trainers-rb raises. Rescue this to catch them all.
  class Error < StandardError; end

  # The model, data or arguments cannot be trained as configured
  class ConfigurationError < Error; end

  # LoRA adapters are missing, cannot be applied, or cannot be loaded
  class AdapterError < Error; end

  # A checkpoint cannot be written or read
  class CheckpointError < Error; end
end
