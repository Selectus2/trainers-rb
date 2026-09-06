# trainers-rb

Fine-tune transformer models in Ruby.

trainers-rb provides a training loop, LoRA (Low-Rank Adaptation), learning rate scheduling, and model serialization for HuggingFace transformer models loaded via [transformers-rb](https://github.com/ankane/transformers-ruby). It builds on [torch-rb](https://github.com/ankane/torch.rb) for autograd, optimizers, and tensor operations.

All the heavy lifting happens in LibTorch C++ kernels. Ruby is the conductor.

## Installation

Add to your Gemfile:

```ruby
gem "trainers-rb"
```

Or install directly:

```bash
gem install trainers-rb
```

### Prerequisites

trainers-rb depends on torch-rb, which requires LibTorch:

```bash
# macOS arm64
curl -L -o /tmp/libtorch.zip https://download.pytorch.org/libtorch/cpu/libtorch-macos-arm64-2.4.0.zip
unzip /tmp/libtorch.zip -d ~/libtorch
bundle config set build.torch-rb --with-torch-dir=$HOME/libtorch/libtorch
```

## Quick Start

```ruby
require "trainers-rb"

# Load a pre-trained model and tokenizer
model, tokenizer = Trainers.from_pretrained(
  "distilbert-base-uncased",
  task: :sequence_classification,
  num_labels: 2
)

# Prepare your dataset
train_data = texts.map.with_index do |text, i|
  encoded = tokenizer.(text, truncation: true, max_length: 128)
  {
    input_ids:      encoded["input_ids"],
    attention_mask: encoded["attention_mask"],
    labels:         labels[i]
  }
end
train_dataset = Trainers::Dataset.new(train_data)

# Configure and train
args = Trainers::TrainingArguments.new(
  output_dir:        "./output",
  num_train_epochs:  3,
  learning_rate:     2e-5,
  eval_strategy:     :epoch
)

trainer = Trainers::Trainer.new(
  model:         model,
  args:          args,
  train_dataset: train_dataset,
  eval_dataset:  val_dataset,
  tokenizer:     tokenizer,
  data_collator: Trainers::DataCollatorWithPadding.new(tokenizer: tokenizer),
  compute_metrics: ->(eval_pred) {
    preds   = eval_pred.predictions.argmax(1)
    correct = preds.eq(eval_pred.label_ids).sum.item
    { accuracy: correct.to_f / eval_pred.label_ids.size(0) }
  }
)

trainer.train
trainer.save_model("./my-model")
```

## LoRA (Parameter-Efficient Fine-Tuning)

Freeze 99% of parameters and train only small low-rank adapter matrices:

```ruby
# Apply LoRA to specific layers
config = Trainers::LoraConfig.new(
  r:              8,         # rank
  lora_alpha:     16,        # scaling factor
  lora_dropout:   0.1,
  target_modules: ["query", "value"],  # which Linear layers to adapt
  bias:           :none      # :none, :all, or :lora_only
)

Trainers::LoraModel.apply(model, config)
# => LoRA applied to 12 modules: ...
# => trainable params: 294,912 || all params: 66,955,010 || trainable%: 0.4404%

# Train as usual
trainer.train

# Save just the adapters (tiny files)
Trainers::LoraModel.save_adapters(model, "./lora-adapters")

# Or merge into base model for inference
Trainers::LoraModel.merge(model)
trainer.save_model("./merged-model")
```

### Loading saved LoRA adapters

```ruby
model, tokenizer = Trainers.from_pretrained("distilbert-base-uncased", num_labels: 2)
Trainers::LoraModel.apply(model, config)
Trainers::LoraModel.load_adapters(model, "./lora-adapters")
```

## Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `output_dir` | `"./output"` | Directory for checkpoints and saved models |
| `num_train_epochs` | `3` | Number of training epochs |
| `per_device_train_batch_size` | `8` | Training batch size |
| `per_device_eval_batch_size` | `8` | Evaluation batch size |
| `learning_rate` | `5e-5` | Peak learning rate for AdamW |
| `weight_decay` | `0.0` | Weight decay (applied to non-bias, non-norm params) |
| `max_grad_norm` | `1.0` | Max gradient norm for clipping |
| `gradient_accumulation_steps` | `1` | Accumulate gradients over N steps |
| `warmup_steps` | `0` | Linear warmup steps |
| `warmup_ratio` | `0.0` | Warmup as fraction of total steps (alternative to warmup_steps) |
| `lr_scheduler_type` | `:linear` | `:linear`, `:cosine`, or `:constant` |
| `eval_strategy` | `:no` | When to evaluate: `:no`, `:epoch`, or `:steps` |
| `eval_steps` | `nil` | Evaluate every N steps (when `eval_strategy: :steps`) |
| `save_strategy` | `:epoch` | When to save: `:no`, `:epoch`, or `:steps` |
| `save_total_limit` | `nil` | Keep only the last N checkpoints |
| `logging_steps` | `500` | Log every N steps |
| `seed` | `42` | Random seed |
| `no_mps` | `false` | Force CPU even if MPS is available |

## Callbacks

Built-in callbacks:

```ruby
# Early stopping
early_stop = Trainers::EarlyStoppingCallback.new(
  patience:    3,
  threshold:   0.01,
  metric_name: "eval_loss"
)

trainer = Trainers::Trainer.new(
  model: model,
  args: args,
  callbacks: [early_stop],
  # ...
)
```

Custom callbacks:

```ruby
class WandbCallback < Trainers::TrainerCallback
  def on_log(args, state, control, logs: nil, **)
    # send logs to Weights & Biases, MLflow, etc.
  end

  def on_evaluate(args, state, control, metrics: nil, **)
    # log evaluation metrics
  end
end
```

### Callback hooks

| Hook | When it fires |
|------|---------------|
| `on_train_begin` | Before the first step |
| `on_train_end` | After the last step |
| `on_epoch_begin` | Start of each epoch |
| `on_epoch_end` | End of each epoch |
| `on_step_begin` | Before each training step |
| `on_step_end` | After each training step |
| `on_log` | When metrics are logged |
| `on_evaluate` | After evaluation |
| `on_save` | After saving a checkpoint |

## Learning Rate Schedulers

Three schedules are available, all with optional linear warmup:

```ruby
# Linear warmup then linear decay to 0 (default)
args = Trainers::TrainingArguments.new(lr_scheduler_type: :linear, warmup_steps: 100)

# Linear warmup then cosine decay to 0
args = Trainers::TrainingArguments.new(lr_scheduler_type: :cosine, warmup_steps: 100)

# Linear warmup then constant
args = Trainers::TrainingArguments.new(lr_scheduler_type: :constant, warmup_steps: 100)
```

## Data Utilities

### Dataset

Wrap an array of hashes:

```ruby
data = [
  { input_ids: [101, 2023, 2003], attention_mask: [1, 1, 1], labels: 1 },
  { input_ids: [101, 2919, 2143], attention_mask: [1, 1, 1], labels: 0 },
]
dataset = Trainers::Dataset.new(data)
```

### Data Collators

Dynamic padding collator (pads each batch to the longest sequence in that batch):

```ruby
collator = Trainers::DataCollatorWithPadding.new(tokenizer: tokenizer)
```

Default collator (no padding, expects uniform-length inputs):

```ruby
collator = Trainers::DefaultDataCollator.new
```

## Supported Tasks

trainers-rb works with any `Torch::NN::Module`. The `Trainers.from_pretrained` convenience method supports these transformers-rb model classes:

| Task | Model class |
|------|-------------|
| `:sequence_classification` | `AutoModelForSequenceClassification` |
| `:token_classification` | `AutoModelForTokenClassification` |
| `:question_answering` | `AutoModelForQuestionAnswering` |

You can also use any custom model:

```ruby
trainer = Trainers::Trainer.new(model: my_custom_model, args: args, ...)
```

## Device Support

trainers-rb auto-detects the best available device:

- **CPU** — always available
- **MPS** — Apple Silicon GPU, used automatically when available

```ruby
# Force CPU
args = Trainers::TrainingArguments.new(no_mps: true)

# Or set explicitly
args = Trainers::TrainingArguments.new(device: Torch.device("mps"))
```

## Architecture

```
trainers-rb
  -> transformers-rb    (model loading, tokenizers, HF Hub)
    -> torch-rb         (autograd, nn modules, optimizers)
    -> tokenizers        (HuggingFace Rust tokenizers via FFI)
    -> safetensors       (weight file I/O)
```

trainers-rb adds the training layer that transformers-rb intentionally omits. Both gems call into the same LibTorch C++ kernels for the actual computation.

## Roadmap

- [ ] More model architectures in transformers-rb (GPT-2, Llama for text generation)
- [ ] Mixed precision training (fp16/bf16)
- [ ] Gradient checkpointing for memory efficiency
- [ ] Dataset streaming for large datasets
- [ ] Distributed training
- [ ] Integration with ONNX export for deployment
- [ ] QLoRA (quantized base model + LoRA)

## Contributing

Bug reports and pull requests are welcome on GitHub. See [CONTRIBUTING.md](CONTRIBUTING.md) for setup, testing, and pull request guidelines.

## License

MIT
