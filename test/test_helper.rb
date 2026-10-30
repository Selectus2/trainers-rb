require "minitest/autorun"
require "tmpdir"
require "trainers-rb"

# Synthetic fixture: a tiny model that never touches the network
class FakeLinear < Torch::NN::Module
  def initialize(in_features, out_features)
    super()
    @weight = Torch.randn(out_features, in_features)
    @bias   = Torch.randn(out_features)
  end

  def forward(x)
    x.mm(@weight.t) + @bias
  end
end

class FakeModel < Torch::NN::Module
  def initialize
    super()
    @fc = FakeLinear.new(10, 2)
  end

  def forward(x)
    @fc.call(x)
  end
end

# Accepts the keyword batch the Trainer passes and returns raw logits,
# so the Trainer computes cross-entropy itself
class TinyClassifier < Torch::NN::Module
  def initialize(in_features = 4, num_labels = 2)
    super()
    @fc = Torch::NN::Linear.new(in_features, num_labels)
  end

  def forward(input:, labels: nil)
    @fc.call(input)
  end
end

# Nested Linear layers named like a transformer attention block, for LoRA
class TinyAttention < Torch::NN::Module
  def initialize(dim)
    super()
    @query = Torch::NN::Linear.new(dim, dim)
    @value = Torch::NN::Linear.new(dim, dim)
  end

  def forward(x)
    @query.call(x) + @value.call(x)
  end
end

class TinyLoraTarget < Torch::NN::Module
  def initialize(dim = 8, num_labels = 2)
    super()
    @attn = TinyAttention.new(dim)
    @out  = Torch::NN::Linear.new(dim, num_labels)
  end

  def forward(x)
    @out.call(@attn.call(x))
  end
end

module TestHelpers
  # Linearly separable 4-feature dataset of `size` rows
  def synthetic_dataset(size = 8)
    rows = Array.new(size) do |i|
      label = i % 2
      { input: Array.new(4) { |j| (label * 2.0 - 1.0) + j * 0.1 }, labels: label }
    end
    Trainers::Dataset.new(rows)
  end

  def training_args(**overrides)
    Trainers::TrainingArguments.new(
      output_dir: @tmpdir || Dir.mktmpdir,
      device: Torch.device("cpu"),
      save_strategy: :no,
      **overrides
    )
  end

  # Runs the block with library output (puts) silenced
  def quietly(&block)
    result = nil
    capture_io { result = block.call }
    result
  end
end

Minitest::Test.include(TestHelpers)
