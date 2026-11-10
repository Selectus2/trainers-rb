# frozen_string_literal: true

module Trainers
  class DataCollatorWithPadding
    attr_reader :tokenizer, :padding, :max_length, :pad_to_multiple_of

    def initialize(tokenizer:, padding: true, max_length: nil, pad_to_multiple_of: nil)
      @tokenizer          = tokenizer
      @padding             = padding
      @max_length          = max_length
      @pad_to_multiple_of  = pad_to_multiple_of
    end

    def call(features)
      return {} if features.empty?

      keys = features.first.keys
      batch = {}

      keys.each do |key|
        values = features.map { |f| f[key] }

        if values.first.is_a?(Array)
          batch[key] = pad_and_stack(key, values)
        elsif values.first.is_a?(Torch::Tensor)
          if values.first.dim == 0
            batch[key] = Torch.stack(values)
          else
            batch[key] = pad_and_stack_tensors(key, values)
          end
        elsif values.first.is_a?(Integer)
          batch[key] = Torch.tensor(values, dtype: :int64)
        elsif values.first.is_a?(Float)
          batch[key] = Torch.tensor(values, dtype: :float32)
        else
          batch[key] = values
        end
      end

      batch
    end

    private

    # Pads, or truncates to max_length, each sequence of Integers
    def pad_and_stack(key, sequences)
      max_len   = compute_max_length(sequences.map(&:length))
      pad_value = pad_value_for(key)

      padded = sequences.map do |seq|
        seq = seq.first(max_len)
        seq + [pad_value] * (max_len - seq.length)
      end

      Torch.tensor(padded, dtype: :int64)
    end

    def pad_and_stack_tensors(key, tensors)
      max_len   = compute_max_length(tensors.map { |t| t.size(0) })
      pad_value = pad_value_for(key)

      padded = tensors.map do |t|
        if t.size(0) < max_len
          padding = Torch.full([max_len - t.size(0)], pad_value, dtype: t.dtype)
          Torch.cat([t, padding])
        else
          t[0...max_len]
        end
      end

      Torch.stack(padded)
    end

    # Labels pad with -100, which cross-entropy ignores
    def pad_value_for(key)
      name = key.to_s
      if name.include?("input_id")
        pad_token_id
      elsif name.include?("label")
        -100
      else
        0
      end
    end

    def compute_max_length(lengths)
      max_len = @max_length || lengths.max
      if @pad_to_multiple_of
        max_len = ((max_len + @pad_to_multiple_of - 1) / @pad_to_multiple_of) * @pad_to_multiple_of
      end
      max_len
    end

    def pad_token_id
      if @tokenizer.respond_to?(:pad_token_id)
        @tokenizer.pad_token_id || 0
      else
        0
      end
    end
  end

  class DefaultDataCollator
    def call(features)
      return {} if features.empty?

      batch = {}
      features.first.keys.each do |key|
        values = features.map { |f| f[key] }

        if values.first.is_a?(Torch::Tensor)
          batch[key] = Torch.stack(values)
        elsif values.first.is_a?(Integer)
          batch[key] = Torch.tensor(values, dtype: :int64)
        elsif values.first.is_a?(Float)
          batch[key] = Torch.tensor(values, dtype: :float32)
        elsif values.first.is_a?(Array)
          batch[key] = Torch.tensor(values)
        else
          batch[key] = values
        end
      end

      batch
    end
  end
end
