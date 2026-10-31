require_relative "test_helper"

class LoraTest < Minitest::Test
  DIM  = 8
  RANK = 4

  def setup
    Torch.manual_seed(0)
    @model = TinyLoraTarget.new(DIM)
    @input = Torch.randn(3, DIM)
    @tmpdir = Dir.mktmpdir
  end

  def teardown
    FileUtils.rm_rf(@tmpdir)
  end

  def apply_lora(model = @model, **config)
    config = { r: RANK, lora_alpha: 8, target_modules: ["query", "value"] }.merge(config)
    quietly { Trainers::LoraModel.apply(model, Trainers::LoraConfig.new(**config)) }
  end

  def lora_modules(model = @model)
    model.named_modules.select { |_, mod| mod.is_a?(Trainers::LoraLinear) }.to_h
  end

  # Give lora_B non-zero values so the adapter actually changes the output
  def perturb_adapters(model = @model)
    lora_modules(model).each_value do |mod|
      mod.load_lora_weights(Torch.randn(RANK, DIM), Torch.randn(DIM, RANK))
    end
  end

  def test_parameter_shapes_and_frozen_base
    apply_lora

    assert_equal %w[attn.query attn.value], lora_modules.keys.sort
    lora_modules.each_value do |mod|
      a = mod.instance_variable_get(:@lora_A)
      b = mod.instance_variable_get(:@lora_B)
      assert_equal [RANK, DIM], a.shape
      assert_equal [DIM, RANK], b.shape
      assert a.requires_grad
      assert b.requires_grad
      assert_in_delta 2.0, mod.scaling, 1e-9 # lora_alpha / r = 8 / 4
    end

    trainable = @model.named_parameters.select { |_, p| p.requires_grad }.keys
    assert_equal %w[attn.query.lora_A attn.query.lora_B attn.value.lora_A attn.value.lora_B], trainable.sort
  end

  def test_initialisation_is_kaiming_a_and_zero_b
    apply_lora
    lora_modules.each_value do |mod|
      assert_equal 0.0, mod.instance_variable_get(:@lora_B).abs.sum.item
      refute_equal 0.0, mod.instance_variable_get(:@lora_A).abs.sum.item
    end
  end

  def test_fresh_adapter_is_identity
    before = Torch.no_grad { @model.call(@input) }
    apply_lora
    after = Torch.no_grad { @model.call(@input) }

    assert Torch.allclose(before, after), "a freshly applied adapter must not change outputs"
  end

  def test_merge_matches_unmerged_forward
    apply_lora
    perturb_adapters
    unmerged = Torch.no_grad { @model.call(@input) }

    quietly { Trainers::LoraModel.merge(@model) }
    merged = Torch.no_grad { @model.call(@input) }

    assert Torch.allclose(unmerged, merged, rtol: 1e-5, atol: 1e-5),
           "max difference #{(unmerged - merged).abs.max.item}"
    assert_empty lora_modules, "merge should replace every LoraLinear with a plain Linear"
  end

  def test_merged_model_has_base_parameter_names
    base_names = @model.state_dict.keys.sort
    apply_lora
    quietly { Trainers::LoraModel.merge(@model) }

    assert_equal base_names, @model.state_dict.keys.sort
  end

  def test_merge_is_idempotent_on_a_layer
    apply_lora
    perturb_adapters
    layer = lora_modules["attn.query"]
    expected = Torch.no_grad { layer.call(@input) }

    2.times { layer.merge! }

    assert layer.merged?
    assert Torch.allclose(expected, Torch.no_grad { layer.call(@input) }, rtol: 1e-5, atol: 1e-5)
  end

  def test_save_and_load_adapters_round_trip
    apply_lora
    perturb_adapters
    config = Trainers::LoraConfig.new(r: RANK, lora_alpha: 8)
    quietly { Trainers::LoraModel.save_adapters(@model, @tmpdir, config: config) }

    assert File.exist?(File.join(@tmpdir, "adapter_model.safetensors"))
    assert_equal RANK, JSON.parse(File.read(File.join(@tmpdir, "lora_config.json")))["r"]

    restored = TinyLoraTarget.new(DIM)
    apply_lora(restored)
    quietly { Trainers::LoraModel.load_adapters(restored, @tmpdir) }

    lora_modules.each do |name, original|
      loaded = lora_modules(restored).fetch(name)
      original.lora_state_dict.each do |key, tensor|
        other = loaded.lora_state_dict.fetch(key)
        assert_equal tensor.dtype, other.dtype, "#{name}.#{key} dtype"
        assert Torch.equal(tensor, other), "#{name}.#{key} values"
      end
    end
  end

  def test_no_matching_target_modules_raises
    error = assert_raises(Trainers::AdapterError) { apply_lora(target_modules: ["q_proj"]) }
    assert_match "q_proj", error.message
  end

  def test_all_linear_targets_every_linear
    apply_lora(target_modules: :all_linear)
    assert_equal %w[attn.query attn.value out], lora_modules.keys.sort
  end

  # Regression: indexes into a ModuleList ("layer.0.query") raised AdapterError
  def test_targets_inside_a_module_list
    model = TinyLayerStack.new(DIM)
    expected = model.call(@input)
    apply_lora(model)

    assert_equal %w[layer.0.query layer.0.value layer.1.query layer.1.value], lora_modules(model).keys.sort
    assert Torch.allclose(expected, model.call(@input), atol: 1e-6)
    assert_equal 4, model.named_parameters.count { |name, p| p.requires_grad && name.include?("lora_") } / 2
  end
end
