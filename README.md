# trainers-rb

[![Gem Version](https://img.shields.io/gem/v/trainers-rb)](https://rubygems.org/gems/trainers-rb)
[![CI](https://github.com/Selectus2/trainers-rb/actions/workflows/ci.yml/badge.svg)](https://github.com/Selectus2/trainers-rb/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE.txt)
[![Ruby](https://img.shields.io/badge/ruby-3.3%20%7C%203.4%20%7C%204.0-red.svg)](.github/workflows/ci.yml)

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

trainers-rb depends on torch-rb, which compiles against LibTorch. Use the LibTorch version that matches your torch-rb: 2.15.0 for torch-rb 0.27, or 2.4.0 for torch-rb 0.17.

```bash
# macOS arm64
curl -L -o /tmp/libtorch.zip https://download.pytorch.org/libtorch/cpu/libtorch-macos-arm64-2.15.0.zip
unzip /tmp/libtorch.zip -d ~/libtorch-2.15.0
bundle config set build.torch-rb --with-torch-dir=$HOME/libtorch-2.15.0/libtorch
```

For Linux, and for running the tests, see [CONTRIBUTING.md](CONTRIBUTING.md).
## Quick Start

```ruby
require "trainers-rb"

# Load a pre-trained model and tokenizer. labels: names the classes in id
# order; they are saved in config.json.
model, tokenizer = Trainers.from_pretrained(
  "distilbert-base-uncased",
  task: :sequence_classification,
  labels: %w[negative positive]
)

# reviews.jsonl: {"text": "A gripping film.", "label": 1} on each line
dataset = Trainers::Dataset.from_jsonl("reviews.jsonl")
                           .map { |row| row.merge(labels: row.delete(:label)) }
                           .tokenize(tokenizer, column: :text, max_length: 128)
train_dataset, val_dataset = dataset.split(test_size: 0.1, seed: 42)

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

## Saving and loading

`save_model` writes a directory that trainers-rb, transformers-rb and Python transformers can all read:

| File | Contents |
|---|---|
| `config.json` | The model's transformers config, with `architectures` and the label maps (`id2label`, `label2id`) |
| `model.safetensors` | The weights |
| `tokenizer.json` | The tokenizer, when one was passed to the `Trainer` |
| `tokenizer_config.json` | The tokenizer class and its special tokens (for transformers-rb tokenizers) |
| `training_args.json` | The `TrainingArguments` used, for your records. Nothing reads it back |

Load it in a fresh process with `Trainers.load_pretrained`. It returns the model in eval mode and the tokenizer (`nil` if none was saved):

```ruby
model, tokenizer = Trainers.load_pretrained("./my-model")
encoded = tokenizer.(["What a great film"], return_tensors: "pt")
model.(input_ids: encoded[:input_ids], attention_mask: encoded[:attention_mask])[:logits]
```

In Python, `AutoModelForSequenceClassification.from_pretrained("./my-model")` and `AutoTokenizer.from_pretrained("./my-model")` load the same directory.

`config.json` is written only for transformers-rb models. A plain `Torch::NN::Module` gets weights only and cannot be loaded with `load_pretrained`. Directories saved by trainers-rb before 0.3.0 have no `config.json`, so `load_pretrained` raises `Trainers::CheckpointError` for them.

A model with unmerged LoRA adapters can't be loaded as a plain model. Merge it first with `LoraModel.merge`, or save only the adapters (see below).

## LoRA (Parameter-Efficient Fine-Tuning)

Freeze the base model and train small low-rank adapter matrices, plus the task head:

```ruby
# Apply LoRA to specific layers
config = Trainers::LoraConfig.new(
  r:              8,         # rank
  lora_alpha:     16,        # scaling factor
  lora_dropout:   0.1,
  # target_modules: ["q_lin", "v_lin"],  # defaults to the attention query and value layers
  bias:           :none,     # :none, :all, or :lora_only
  task_type:      :sequence_classification
  # modules_to_save: ["classifier"]  # defaults to the task head for task_type
)

Trainers::LoraModel.apply(model, config)
# => Training in full: pre_classifier, classifier
# => LoRA target_modules ["q_lin", "v_lin"] (default) applied to 12 modules: ...
# => trainable params: 739,586 || all params: 67,102,466 || trainable%: 1.1022%

# Train as usual
trainer.train

# Save just the adapters and the task head (small files)
Trainers::LoraModel.save_adapters(model, "./lora-adapters")

# Or merge into base model for inference
Trainers::LoraModel.merge(model)
trainer.save_model("./merged-model")
```

The task head is trained in full, as in PEFT. `from_pretrained` initialises it randomly, so it has to learn along with the adapters. `modules_to_save` defaults to the head for `task_type`:

| `task_type` | Kept trainable |
|---|---|
| `:sequence_classification` | `pre_classifier`, `classifier`, `score` |
| `:token_classification` | `classifier` |
| `:question_answering` | `qa_outputs` |

Names match the last part of a module's name. Pass `modules_to_save: []` to freeze the head, or your own list for other models.

`target_modules` defaults to the attention query and value layers of the model's `config.model_type`:

| `model_type` | Default `target_modules` |
|---|---|
| `bert`, `xlm-roberta` | `query`, `value` |
| `distilbert` | `q_lin`, `v_lin` |
| `deberta-v2` | `query_proj`, `value_proj` |
| `mpnet` | `q`, `v` |

Any other `model_type` needs `target_modules:`. Models without a transformers config default to `query` and `value`. Pass `:all_linear` to adapt every Linear layer.

A target matches a module when it is the module's full name or the last part of it, as in PEFT: `"query"` matches `attention.self.query` but not `pos_query_proj`. A target that matches nothing that way still matches by substring, with a deprecation warning. Substring matching ends in 0.5.0.

### Loading saved LoRA adapters

```ruby
model, tokenizer = Trainers.from_pretrained("distilbert-base-uncased", num_labels: 2)
Trainers::LoraModel.apply(model, config)
Trainers::LoraModel.load_adapters(model, "./lora-adapters")
```

Use the same base model and `LoraConfig` as in training. The adapter file holds every trained weight, including the task head, so the loaded model predicts exactly as the trained one did.

## Checkpoints and resuming

With `save_strategy: :steps` or `:epoch`, the trainer writes `checkpoint-N` directories in `output_dir`, where N is the step. Each one can resume training:

```ruby
args = Trainers::TrainingArguments.new(
  output_dir: "./output",
  resume_from_checkpoint: true   # the latest checkpoint-N, or pass a directory
)
Trainers::Trainer.new(model: model, args: args, ...).train
```

| File | Full fine-tune | LoRA run |
|---|---|---|
| `model.safetensors`, `config.json`, tokenizer files | yes | no |
| `adapter_model.safetensors` (adapters and task head) | no | yes |
| `optimizer.safetensors` (AdamW moments for the trainable weights) | yes | yes |
| `trainer_state.json` (step, epoch position, seed, log history) | yes | yes |

A LoRA checkpoint doesn't hold the frozen base model. On DistilBERT it is about 9 MB, against 268 MB for the full weights. To resume a LoRA run, load the same base model and apply the same `LoraConfig` before calling `train`.

Resuming restores the weights, the optimizer, the learning-rate schedule, the step count and the data order. The rest of the run uses the same batches and learning rates as an uninterrupted one. With `dropout` the results still differ slightly: torch-rb can't save Torch's random-number state, so the trainer reseeds it from `seed` and the step, and dropout masks after the resume differ from the original run's. Without dropout the resumed run matches exactly.

Resume with the same `seed`, dataset and arguments as the original run. The trainer logs a warning when the seed or the total step count differs. Checkpoints written before 0.3.0 hold weights only; resuming from one raises `Trainers::CheckpointError`.

## Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `output_dir` | `"./output"` | Directory for checkpoints and saved models |
| `num_train_epochs` | `3` | Number of training epochs |
| `max_steps` | `nil` | Number of optimizer steps to train for. Overrides `num_train_epochs` when set |
| `per_device_train_batch_size` | `8` | Training batch size |
| `per_device_eval_batch_size` | `8` | Evaluation batch size |
| `learning_rate` | `5e-5` | Peak learning rate for AdamW |
| `weight_decay` | `0.0` | Weight decay (applied to non-bias, non-norm params) |
| `max_grad_norm` | `1.0` | Max gradient norm for clipping |
| `gradient_accumulation_steps` | `1` | Accumulate gradients over N batches per optimizer step |
| `warmup_steps` | `0` | Linear warmup steps |
| `warmup_ratio` | `0.0` | Warmup as fraction of total steps (alternative to warmup_steps) |
| `lr_scheduler_type` | `:linear` | `:linear`, `:cosine`, or `:constant` |
| `eval_strategy` | `:no` | When to evaluate: `:no`, `:epoch`, or `:steps` |
| `eval_steps` | `nil` | Evaluate every N steps (when `eval_strategy: :steps`) |
| `save_strategy` | `:epoch` | When to save: `:no`, `:epoch`, or `:steps` |
| `save_total_limit` | `nil` | Keep only the last N checkpoints |
| `save_steps` | `500` | Save a checkpoint every N steps (when `save_strategy: :steps`) |
| `resume_from_checkpoint` | `nil` | A checkpoint directory to resume from, or `true` for the latest in `output_dir` |
| `logging_steps` | `500` | Log every N steps |
| `seed` | `42` | Seeds Torch, Ruby's RNG and data shuffling at the start of `train`; `nil` disables seeding |
| `dataloader_drop_last` | `false` | Skip the final batch when it is smaller than the batch size |
| `label_names` | `["labels"]` | Batch keys that hold labels; the first is used for the built-in cross-entropy loss |
| `no_mps` | `false` | Force CPU even if MPS is available |

**A step is one optimizer update.** With `gradient_accumulation_steps: 4`, one step covers four batches, and `max_steps`, `global_step`, `logging_steps`, `eval_steps`, `save_steps`, `warmup_steps` and the learning-rate schedule all count optimizer updates. When an epoch's batch count is not a multiple of the accumulation steps, the last, shorter window still makes an update, with its loss averaged over the batches it contains, so no gradient carries into the next epoch.

Invalid values, such as a negative learning rate, a zero batch size or an unknown `lr_scheduler_type`, raise `ArgumentError` when `TrainingArguments` is created.

## Reproducibility

`Trainer#train` seeds Torch, Ruby's global RNG and its own shuffling from `args.seed`. The model's initial weights are created before training starts, so seed those yourself:

```ruby
Trainers.set_seed(42)
model, tokenizer = Trainers.from_pretrained("distilbert-base-uncased", num_labels: 2)
```

Two runs with the same seed, data and arguments produce identical loss curves on CPU. On Apple Silicon (MPS), some kernels are not deterministic, so expect small differences between runs.

## Logging

trainers-rb reports progress through `Trainers.logger`, which writes to `$stderr` at `INFO` by default. Nothing is written to stdout.

```ruby
Trainers.logger = Rails.logger                  # use your app's logger
Trainers.logger.level = Logger::WARN            # only warnings and errors
Trainers.logger = nil                           # completely silent
```

To print progress to stdout instead, add `Trainers::PrinterCallback.new` to `callbacks:`.

## Errors

Every error trainers-rb raises descends from `Trainers::Error`:

| Error | Raised when |
|---|---|
| `Trainers::ConfigurationError` | The model has no trainable parameters, returns no loss and no labels are present, or the training data yields no batches |
| `Trainers::AdapterError` | No module matches `target_modules`, an adapter file is missing, or `save_adapters` finds no adapters |
| `Trainers::CheckpointError` | Reserved for checkpoint save and load errors |

```ruby
begin
  trainer.train
rescue Trainers::Error => e
  # anything trainers-rb raised
end
```

Invalid `TrainingArguments` values raise `ArgumentError`.

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

Rows are Hashes with Symbol keys. Load them from a file, or wrap an Array:

```ruby
dataset = Trainers::Dataset.from_jsonl("train.jsonl")      # one JSON object per line
dataset = Trainers::Dataset.from_csv("train.csv")          # header row gives the keys
dataset = Trainers::Dataset.from_csv("train.tsv", col_sep: "\t", converters: :numeric)
dataset = Trainers::Dataset.new([{ text: "A gripping film.", labels: 1 }])
```

CSV values are Strings unless you pass `converters: :numeric`. Other options go to Ruby's `CSV`.

```ruby
# A new Dataset with each row replaced. A block that doesn't return Hashes
# gives an Array, as Enumerable#map does.
dataset = dataset.map { |row| row.merge(labels: Integer(row[:label])) }
dataset = dataset.filter { |row| row[:text].size > 10 }  # select returns an Array

# Shuffled with Random.new(seed): the same seed gives the same split
train, test = dataset.split(test_size: 0.2, seed: 42)    # or a row count: test_size: 500

# input_ids and attention_mask, truncated to max_length tokens. The text
# column is dropped unless keep: true.
dataset = dataset.tokenize(tokenizer, column: :text, max_length: 256)
dataset = dataset.tokenize(tokenizer, column: [:premise, :hypothesis])  # text pairs
```

`tokenize` takes a transformers-rb tokenizer or a `Tokenizers::Tokenizer`.

### Data Collators

Dynamic padding collator (pads each batch to the longest sequence in that batch):

```ruby
collator = Trainers::DataCollatorWithPadding.new(tokenizer: tokenizer)
```

Default collator (no padding, expects uniform-length inputs):

```ruby
collator = Trainers::DefaultDataCollator.new
```

## Support

### Models and tasks

trainers-rb trains any `Torch::NN::Module` whose `forward` takes the batch's keys. `Trainers.from_pretrained` loads transformers-rb models through its `AutoModelFor*` classes, which cover these architectures:

<!-- architectures:start -->
| `model_type` | Sequence classification | Token classification | Question answering | Default LoRA targets |
|---|---|---|---|---|
| `bert` | ✓ | ✓ | – | `query`, `value` |
| `deberta-v2` | ✓ | – | – | `query_proj`, `value_proj` |
| `distilbert` | ✓ | – | ✗ no loss | `q_lin`, `v_lin` |
| `mpnet` | – | – | – | `q`, `v` |
| `xlm-roberta` | ✓ | – | – | `query`, `value` |
<!-- architectures:end -->

This table is checked against transformers-rb by the test suite. **Question answering can't be trained yet:** transformers-rb 0.2.0's `DistilBertForQuestionAnswering` raises `Todo` when given `start_positions`, so there is no loss. transformers-rb also has heads that its `AutoModelFor*` classes don't reach, such as `MPNetForSequenceClassification`. Build those directly with their config and pass them to `Trainer`:

```ruby
trainer = Trainers::Trainer.new(model: my_custom_model, args: args, ...)
```

### Platforms

| | Tested in CI | Hand-tested | Not supported |
|---|---|---|---|
| Ruby | 3.3, 3.4, 4.0 | | 3.2 and older |
| torch-rb | 0.17.1, 0.27 | | older than 0.17.1 |
| OS | Ubuntu 24.04, macOS 15 (Apple Silicon) | macOS 27 | Windows (untested) |
| CPU | ✓ | ✓ | |
| MPS (Apple Silicon GPU) | | DistilBERT-base with LoRA, Apple M3 Pro | |
| CUDA (NVIDIA GPU) | | | Untested: chosen automatically, but nobody has run it yet |
| Multiple GPUs | | | Not planned |

Reports for other hardware are welcome as issues, with the "Training on" log line.

### Devices

trainers-rb picks the first available device, in this order:

- **CUDA**: NVIDIA GPU
- **MPS**: Apple Silicon GPU, skipped with `no_mps: true`
- **CPU**: always available

The first log line of a run shows what it runs on. For DistilBERT with LoRA on an Apple M3 Pro:

```
Training on mps (float32): 9,596 examples, 1 epoch, 600 optimizer steps (batch 16 x accumulation 1), 739,586 of 67,102,466 parameters trainable (1.10%)
```

That run, on 10,662 movie-review sentences ([rotten_tomatoes](https://www.cs.cornell.edu/people/pabo/movie-review-data/)) with 1,066 held out, reached 82.8% accuracy in 74 seconds. A full fine-tune reached 83.6% in 121 seconds.

```ruby
# Force CPU
args = Trainers::TrainingArguments.new(device: Torch.device("cpu"))

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

What comes next is chosen from real fine-tunes. If something was missing in yours, [open an issue](https://github.com/Selectus2/trainers-rb/issues): a report from a user outranks this list. Candidates are collected in [#20](https://github.com/Selectus2/trainers-rb/issues/20).

**Not planned, at any version:** custom kernels, quantisation (including QLoRA), multi-GPU and distributed training, RL methods (GRPO, DPO), vision or speech models, GGUF export. torch-rb can't reach most of them from Ruby.

## Contributing

Bug reports and pull requests are welcome. [CONTRIBUTING.md](CONTRIBUTING.md) covers LibTorch setup, running the tests against each supported torch-rb, and what a good PR looks like. Report security issues privately, as described in [SECURITY.md](SECURITY.md).

## License

MIT
