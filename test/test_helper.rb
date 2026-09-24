require "minitest/autorun"
require "trainers-rb"

# Synthetic fixture: a tiny model that never touches the network or LibTorch
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
