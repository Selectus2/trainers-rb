# Changelog

## 0.1.2 (unreleased)

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
