# Changelog

## 0.3.0 (2026-11-10)

**This release changes LoRA training results.** `LoraModel.apply` now keeps the task head trainable, as PEFT does. It used to freeze it at its random initial weights. Expect better LoRA models than 0.2.0 with the same arguments. Full fine-tunes are unchanged.

### Added

- **Checkpoints resume training** (#18). Set `resume_from_checkpoint:` to a checkpoint directory, or to `true` for the latest `checkpoint-N` in `output_dir`. Each checkpoint now writes `optimizer.safetensors` (AdamW moments, by parameter name) and `trainer_state.json` (step, epoch position, seed, log history and format version). Resuming restores the weights, the optimizer, the learning-rate schedule and the data order, then skips the batches already trained, without collating them. Without dropout, a resumed run matches an uninterrupted one exactly. With dropout it matches closely but not exactly: torch-rb can't save Torch's random state, so it is reseeded from `seed` and the step. Resuming from a checkpoint written before 0.3.0, or of an unknown format, raises `CheckpointError`.
- **`Trainers.load_pretrained(dir)`** returns `[model, tokenizer]` from a directory written by `save_model` (#17). The model is in eval mode on the CPU. It raises `CheckpointError` when the directory has no `config.json`, or when the weights don't match the config. It raises `AdapterError` when the weights hold unmerged LoRA adapters.
- `LoraConfig#modules_to_save`: modules trained in full alongside the adapters. It defaults to the task head for `task_type`, for example `pre_classifier`, `classifier` and `score` for `:sequence_classification`. Pass `[]` to keep the old behaviour.

### Fixed

- **LoRA checkpoints hold the adapters, not the full model** (#19). On DistilBERT-base, a checkpoint was 268 MB of `model.safetensors` and couldn't resume. It is now 3.0 MB of adapters and task head plus 5.9 MB of optimizer state, and it resumes.
- **`LoraModel.apply` no longer freezes the task head.** `from_pretrained` initialises the head randomly, and it stayed that way through training. `task_type` was stored but never used.
- **Saved adapters reproduce the trained model.** `save_adapters` writes every trainable weight, including the task head, and `load_adapters` restores it. Before, a reloaded model got a fresh random head and predicted differently from the one that was trained. `load_adapters` raises `AdapterError` for weights the model has no place for, instead of silently skipping them.
- **`save_model` writes `config.json`** (#17), with `architectures` and the label maps, so the directory can be loaded again. Python transformers loads it too: `AutoModelForSequenceClassification.from_pretrained(dir)` gives logits within 3e-8 of Ruby's.
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
