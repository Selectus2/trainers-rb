# frozen_string_literal: true

module Trainers
  # Wraps a frozen Torch::NN::Linear with low-rank A/B adapter matrices.
  #
  # Forward: y = Wx + b + (x @ A^T @ B^T) * scaling
  #
  # Only lora_A and lora_B are trainable. The original weight and bias
  # are frozen, keeping ~99% of parameters fixed during fine-tuning.
  class LoraLinear < Torch::NN::Module
    attr_reader :in_features, :out_features, :r, :scaling

    def initialize(original_linear, r:, lora_alpha:, lora_dropout: 0.0)
      super()

      @in_features  = original_linear.instance_variable_get(:@in_features) ||
                      original_linear.weight.size(1)
      @out_features = original_linear.instance_variable_get(:@out_features) ||
                      original_linear.weight.size(0)
      @r            = r
      @scaling      = lora_alpha.to_f / r

      # Freeze original parameters
      @weight = original_linear.weight
      @weight.requires_grad = false

      @bias = original_linear.bias
      @bias.requires_grad = false if @bias

      # LoRA low-rank matrices (these are the only trainable parameters)
      # A: (r, in_features)  — initialized with Kaiming uniform
      # B: (out_features, r) — initialized to zero so LoRA starts as identity
      @lora_A = Torch::NN::Parameter.new(Torch.empty(@r, @in_features))
      Torch::NN::Init.kaiming_uniform!(@lora_A, a: Math.sqrt(5))

      @lora_B = Torch::NN::Parameter.new(Torch.zeros(@out_features, @r))

      @lora_dropout = lora_dropout > 0 ? Torch::NN::Dropout.new(p: lora_dropout) : nil
      @merged       = false
    end

    def merged?
      @merged
    end

    def forward(x)
      # Original linear
      base_output = Torch::NN::F.linear(x, @weight, @bias)

      # After merge! the adapter is already folded into @weight
      return base_output if @merged

      # LoRA path
      lora_input = @lora_dropout ? @lora_dropout.call(x) : x
      lora_output = lora_input.matmul(@lora_A.t).matmul(@lora_B.t) * @scaling

      base_output + lora_output
    end

    # Merge LoRA weights into the base weight matrix (for inference).
    # Idempotent: merging twice would add the adapter twice.
    def merge!
      return self if @merged

      Torch.no_grad do
        @weight.add!(@lora_B.matmul(@lora_A) * @scaling)
      end
      @merged = true
      self
    end

    # A plain Linear holding the merged weights, so a merged model has the
    # same parameter names as the base model
    def to_linear
      merge!
      linear = Torch::NN::Linear.new(@in_features, @out_features, bias: !@bias.nil?)
      Torch.no_grad do
        linear.weight.copy!(@weight)
        linear.bias.copy!(@bias) if @bias
      end
      linear
    end

    # Extract LoRA adapter state for saving
    def lora_state_dict
      { "lora_A" => @lora_A.data, "lora_B" => @lora_B.data }
    end

    # Load LoRA weights from saved state
    def load_lora_weights(lora_a_tensor, lora_b_tensor)
      Torch.no_grad do
        @lora_A.copy!(lora_a_tensor)
        @lora_B.copy!(lora_b_tensor)
      end
    end

    def extra_repr
      "in_features=#{@in_features}, out_features=#{@out_features}, " \
      "r=#{@r}, scaling=#{@scaling}"
    end
  end
end
