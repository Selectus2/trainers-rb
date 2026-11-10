# Changelog

## 0.4.0 (2026-11-10)

**Upgrade from 0.3.0 if you train on MPS or CUDA.** 0.3.0 crashes when `evaluate` runs before `train`, and can run out of GPU memory during evaluation. Both are fixed here (see Fixed).

**This release changes LoRA defaults.** `LoraConfig#target_modules` now defaults to the query and value layers of the model's architecture, and targets match by name parts rather than substrings. See Changed.

### Added

- **Datasets from files** (#30): `Dataset.from_jsonl(path)` and `Dataset.from_csv(path, **csv_options)`. A malformed JSONL line raises `ConfigurationError` naming the file and line.
- **`Dataset#split(test_size:, seed: 42)`** returns `[train, test]`, shuffled with `Random.new(seed)`. `test_size` is a fraction or a row count.
- **`Dataset#map` and `Dataset#filter`** return new Datasets, and `map` passes the block a copy of each row, so `row.delete(:label)` leaves the source Dataset unchanged. `map` returns an Array, as before, when the block doesn't return Hashes. `select` still returns an Array.
- **`Dataset#tokenize(tokenizer, column: :text, max_length: 512, keep: false)`** (#31) adds `input_ids` and `attention_mask` (and `token_type_ids` for text pairs), truncated to `max_length`. It takes a transformers-rb tokenizer or a `Tokenizers::Tokenizer`, and restores the latter's truncation setting afterwards.
- **`Trainers.from_pretrained(labels:)`** (#32) sets `num_labels`, `id2label` and `label2id`, so `config.json`, `load_pretrained` and Python see the class names. A `num_labels:` that contradicts `labels:` raises `ArgumentError`.
- **A log line when training starts** (#35) with the device, dtype, number of examples, epochs, optimizer steps, batch and accumulation, and how many parameters are trainable.
- `from_pretrained(task: :question_answering)` warns that transformers-rb 0.2.0 can't compute the question-answering loss.
- **`max_steps:`** (#34) trains for that many optimizer steps, overriding `num_train_epochs`. Epochs continue until the budget is spent, including when callbacks end some early with `should_epoch_stop`. The learning-rate schedule and `warmup_ratio` use it as the total.

### Fixed

- **Token classification evaluates and predicts** (#27). `evaluate` crashed on per-token logits `[batch, seq, labels]`, and `predict` crashed when batches were padded to different lengths. Both now pad per-batch results to the longest batch with `-100` before joining them, so `compute_metrics` gets predictions and labels of matching shape. Training BERT for token classification also failed: transformers-rb 0.2.0's `BertForTokenClassification` raises `NameError` (`CrossEntropyLoss`) whenever it gets labels. The Trainer now runs it without labels and computes the loss itself.
- **Models whose `forward` takes no label keyword train and evaluate again.** `evaluate` passed the labels in and raised `ArgumentError: unknown keyword: :labels`. The Trainer now retries without them and computes the loss itself. Training gets the same fallback.
- **`evaluate` uses the model's own loss** when it returns one, instead of always recomputing cross-entropy over the logits. The fallback cross-entropy flattens per-token logits and ignores `-100` labels.
- **`eval_loss` is averaged over rows, not batches.** A short last batch used to count as much as a full one, so `eval_loss` can differ slightly from 0.3.0 when the eval set isn't a multiple of `per_device_eval_batch_size`.
- **CUDA is used when available** (#28). `resolved_device` only ever chose MPS or the CPU. The order is now an explicit `device:`, then CUDA, then MPS, then the CPU.
- **`DataCollatorWithPadding` truncates rows longer than `max_length`** (#29). Rows given as Arrays raised `ArgumentError: negative argument`. Rows given as Tensors were already truncated.
- **`DataCollatorWithPadding` pads Tensor labels with `-100`**, the same as Array labels. They used to be padded with `0`, a real class, so padding counted towards the loss.
- **Memory stays flat during training and evaluation.** torch-rb tensors are freed only when Ruby garbage-collects them, and Ruby can't see the memory LibTorch holds, so it rarely collects. Activations piled up batch after batch: evaluating DistilBERT on MPS ran out of memory at 22.6 GiB within 17 batches. The Trainer now runs a minor garbage collection after each batch, which kept that evaluation at 0.8 GB and costs under a millisecond per batch.
- **`evaluate` and `predict` move the model to the training device.** Only `train` did, so calling `evaluate` before `train` on MPS or CUDA, for a baseline, crashed with `Passed CPU tensor to MPS op`. Found by fine-tuning DistilBERT on MPS.
- `predict` no longer passes labels to the model, and puts the model back in training mode afterwards, as `evaluate` does.

### Changed

- **LoRA `target_modules` defaults per architecture** (#33). It was `["query", "value"]`, which matched nothing on DistilBERT. The default is now `nil`, meaning `q_lin`/`v_lin` for DistilBERT, `query`/`value` for BERT and XLM-RoBERTa, `query_proj`/`value_proj` for DeBERTa-v2 and `q`/`v` for MPNet. Other transformers models need `target_modules:`, and models without a config keep `["query", "value"]`.
- **LoRA targets match whole name parts**, as in PEFT. `"query"` matches `attention.self.query` but no longer `pos_query_proj`. A target that only matches by substring still works, with a deprecation warning; substring matching ends in 0.5.0.
- **README "Support" section** (#35) replaces "Supported Tasks" and "Device Support". It has a table of architectures and tasks that the test suite checks against transformers-rb, and a platform matrix that separates tested in CI, hand-tested and unsupported. Question answering is now listed as not trainable. It was listed as supported before.
- New runtime dependency: `csv`, which is no longer a default gem from Ruby 3.4.

## 0.3.0 (2026-11-10)

**This release changes LoRA training results.** `LoraModel.apply` now keeps the task head trainable, as PEFT does. It used to freeze it at its random initial weights. Expect better LoRA models than 0.2.0 with the same arguments. Full fine-tunes are unchanged.

### Added

- **Checkpoints resume training** (#18). Set `resume_from_checkpoint:` to a checkpoint directory, or to `true` for the latest `checkpoint-N` in `output_dir`. Each checkpoint now writes `optimizer.safetensors` (AdamW moments, by parameter name) and `trainer_state.json` (step, epoch position, seed, log history and format version). Resuming restores the weights, the optimizer, the learning-rate schedule and the data order, then skips the batches already trained, without collating them. Without dropout, a resumed run matches an uninterrupted one exactly. With dropout it matches closely but not exactly: torch-rb can't save Torch's random state, so it is reseeded from `seed` and the step. Resuming from a checkpoint written before 0.3.0, or of an unknown format, raises `CheckpointError`. Checkpoints are written to a hidden `.checkpoint-N.tmp` directory and renamed into place once complete, so a run killed mid-save resumes from the previous checkpoint. When a callback stops training part-way through an epoch, an epoch-end checkpoint records the batches done, so resuming continues from there. An epoch ended early with `should_epoch_stop` is recorded as complete, so resuming skips its remaining batches, as the uninterrupted run did.
- **`Trainers.load_pretrained(dir)`** returns `[model, tokenizer]` from a directory written by `save_model` (#17). The model is in eval mode on the CPU. It raises `CheckpointError` when the directory has no `config.json`, or when the weights don't match the config. It raises `AdapterError` when the weights hold unmerged LoRA adapters.
- `LoraConfig#modules_to_save`: modules trained in full alongside the adapters. It defaults to the task head for `task_type`, for example `pre_classifier`, `classifier` and `score` for `:sequence_classification`. Pass `[]` to keep the old behaviour.

### Fixed

- **LoRA checkpoints hold the adapters, not the full model** (#19). On DistilBERT-base, a checkpoint was 268 MB of `model.safetensors` and couldn't resume. It is now 3.0 MB of adapters and task head plus 5.9 MB of optimizer state, and it resumes.
- **`LoraModel.apply` no longer freezes the task head.** `from_pretrained` initialises the head randomly, and it stayed that way through training. `task_type` was stored but never used.
- **Saved adapters reproduce the trained model.** `save_adapters` writes every trainable weight, including the task head, and `load_adapters` restores it. Before, a reloaded model got a fresh random head and predicted differently from the one that was trained. `load_adapters` raises `AdapterError` for weights the model has no place for, instead of silently skipping them.
- **`save_model` writes `config.json`** (#17), with the label maps and `architectures` set to the class being saved, so the directory loads back as that class even when the pretrained config named another head. Python transformers loads it too: `AutoModelForSequenceClassification.from_pretrained(dir)` gives logits within 3e-8 of Ruby's.
- **`save_model` writes the tokenizer for transformers-rb tokenizers** (#17). It used to save `tokenizer.json` only for objects that respond to `save`, which ruled out `AutoTokenizer` results such as `DistilBertTokenizerFast`, so no tokenizer was ever written. It now saves the backend as `tokenizer.json`, plus `tokenizer_config.json` with the tokenizer class and special tokens.
- `model.safetensors` and `adapter_model.safetensors` carry the `"format" => "pt"` metadata that the transformers loaders in Ruby and Python check for.


### Changed

- README: the LoRA example uses DistilBERT's module names (`q_lin`, `v_lin`) and shows its real output. The roadmap lists what is not planned, and "Known limitations" is gone, because its three issues are fixed.

## 0.2.0 (2026-10-31)

**This release changes training results.** Runs that use `gradient_accumulation_steps > 1` now follow the intended learning-rate schedule. All runs now apply `seed:`, so data order is different from 0.1.x and reproducible. Expect different (better) final models than 0.1.x with the same arguments.


### Changed

- **A step is now one optimizer update** (#12). `global_step`, `max_steps`, `logging_steps`, `eval_steps`, `save_steps`, `warmup_steps` and the scheduler all count optimizer updates, not batches. The scheduler is sized as `ceil(batches_per_epoch / gradient_accumulation_steps) * num_train_epochs`, so it now decays to the end. Previously, with accumulation 4, a linear schedule ended near 75% of the peak learning rate.
  - The last, shorter accumulation window of an epoch now makes an optimizer update, with its loss averaged over the batches it holds. Previously its gradients leaked into the next epoch.
  - `on_step_begin` fires at the start of each accumulation window and `on_step_end` after each optimizer update. Nothing changes with `gradient_accumulation_steps: 1`.
- **`seed:` is applied** (#13). `train` seeds Torch, Ruby's RNG and its own shuffling. `seed: nil` disables this. New `Trainers.set_seed(seed)` seeds model initialisation.
- **Progress goes to `Trainers.logger`** on `$stderr`, not stdout (#15). Set `Trainers.logger = nil` for a silent library. The default callback is now `Trainers::LoggingCallback`; `PrinterCallback` still prints to stdout if you add it to `callbacks:`.
- **Errors descend from `Trainers::Error`** (#16): `ConfigurationError`, `AdapterError` and `CheckpointError`. Code that rescued `RuntimeError` for these must rescue `Trainers::Error` instead.
- `SaveUtils.save_lora_adapters` and `LoraModel.save_adapters` raise `AdapterError` when the model has no adapters, instead of printing and returning `nil` (#16).
- **Requires Ruby 3.3 or newer.** Ruby 3.1 and 3.2 are end-of-life.
- New runtime dependency: `logger`, which Ruby 3.5+ no longer ships as a default gem.

### Fixed

- `dataloader_drop_last: true` skips the final partial batch in training, evaluation and prediction (#14). `train` raises `ConfigurationError` if that leaves no batches.
- `label_names:` is honoured (#14). Every listed key is passed to the model, and the first is used for the built-in cross-entropy loss and `evaluate`.
- `LoraModel.apply` works on models whose layers sit in a `ModuleList`, which covers every HuggingFace transformer, for example DistilBERT's `transformer.layer.0.attention.q_lin`. It used to raise `Could not find module part '0'`, because `ModuleList` stores its children in a Hash keyed by `"0"`, not in an Array.
- When an epoch's last optimizer step was already logged, the end-of-epoch log no longer repeats it.
- Learning rates and other floats below 0.001 are logged in scientific notation (`2.000e-05`), not as `0.0000`.

### Added

- `TrainingArguments` validates `seed` (nil or a non-negative Integer) and `label_names` (an Array of Strings or Symbols).

## 0.1.2 (2026-10-30)

### Fixed

- `LoraModel.merge` no longer applies adapters twice. Merged layers kept adding the LoRA path on top of the merged weights, so a merged model's outputs were wrong. Each `LoraLinear` is now replaced by a plain `Torch::NN::Linear` holding the merged weights, so a saved merged model has the base model's parameter names. `LoraLinear#merge!` is idempotent and adds `#merged?` and `#to_linear`.
- `Trainers::VERSION` now matches the gem version. The gemspec reads it from `lib/trainers/version.rb`. (#2)


### Changed

- `TrainingArguments` validates its values at construction, and raises `ArgumentError` naming the argument, the value received and what was expected (#8):
  - positive: `learning_rate`
  - positive Integer: `num_train_epochs`, both batch sizes, `gradient_accumulation_steps`, `logging_steps`, `save_steps`
  - non-negative: `weight_decay`, `max_grad_norm`; `warmup_steps` must also be an Integer
  - `warmup_ratio` must be within 0..1, and `eval_steps` must be nil or a positive Integer
  - `lr_scheduler_type` must be one of `:linear`, `:cosine`, `:constant`; `eval_strategy` and `save_strategy` one of `:no`, `:epoch`, `:steps`

  Previously these were accepted and failed later, or were silently ignored. Defaults are unchanged.

### Internal

- CI on Ruby 3.3, 3.4 and 4.0 against torch-rb 0.17 (LibTorch 2.4.0) and 0.27 (LibTorch 2.15.0), using Appraisal gemfiles, plus macOS arm64; every job is required (#9)
- Tests for LR schedulers and warmup (#6), callback order and arguments (#7), LoRA shapes, identity init, merge and adapter round-trip (#5), and a public API surface test (#10)
- Development dependency minitest bumped to ~> 6.0
- CONTRIBUTING.md with LibTorch setup, SECURITY.md, README badges and a known-limitations section (#11)

## 0.1.1 (2026-06-09)

- Fix gemspec homepage, source and changelog URLs to point to github.com/Selectus2/trainers-rb
- No code changes. The published gem reports `Trainers::VERSION` as "0.1.0"; the constant is corrected here

## 0.1.0 (2026-06-07)

- Initial release
- Trainer class with train/evaluate/predict
- TrainingArguments configuration
- AdamW optimizer with weight decay param groups
- Linear, cosine, and constant learning rate schedulers with warmup
- LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning
- LoRA adapter save/load via safetensors
- LoRA merge for inference deployment
- DataCollatorWithPadding for dynamic batch padding
- Callback system with PrinterCallback and EarlyStoppingCallback
- Model saving via safetensors
- CPU and MPS (Apple Silicon) device support
- Gradient accumulation and gradient clipping
