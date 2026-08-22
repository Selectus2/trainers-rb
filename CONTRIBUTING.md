# Contributing to trainers-rb

Thank you for your interest in contributing! trainers-rb is a Ruby gem for fine-tuning HuggingFace transformer models. It provides a training loop, LoRA (Low-Rank Adaptation), learning rate scheduling, callbacks, and model serialization on top of [transformers-rb](https://github.com/ankane/transformers-ruby) and [torch-rb](https://github.com/ankane/torch.rb).

Bug reports, feature ideas, and pull requests are welcome on [GitHub](https://github.com/trainers-rb/trainers-rb).

## Development prerequisites

| Requirement | Version / notes |
|-------------|-----------------|
| **Ruby** | `>= 3.1.0` (see `trainers-rb.gemspec`) |
| **Bundler** | Required to install dependencies from the `Gemfile` |
| **LibTorch** | Required by `torch-rb` for tensor operations and autograd |
| **torch-rb** | `>= 0.17.1` |
| **transformers-rb** | `>= 0.2.0` |
| **safetensors** | `>= 0.1.1` (model weight I/O) |
| **tokenizers** | `>= 0.5.3` (HuggingFace tokenizers via FFI) |

Development-only gems (installed via Bundler):

| Gem | Version |
|-----|---------|
| **rake** | `~> 13.0` |
| **minitest** | `~> 5.0` |

### LibTorch setup

`torch-rb` links against a local LibTorch installation at build time. The README documents one setup path (macOS arm64, LibTorch 2.4.0 CPU):

```bash
# macOS arm64 (from README)
curl -L -o /tmp/libtorch.zip https://download.pytorch.org/libtorch/cpu/libtorch-macos-arm64-2.4.0.zip
unzip /tmp/libtorch.zip -d ~/libtorch
bundle config set build.torch-rb --with-torch-dir=$HOME/libtorch/libtorch
```

For other platforms, download the matching LibTorch build from [pytorch.org](https://pytorch.org/get-started/locally/) and point Bundler at it with the same `build.torch-rb` config. Refer to the [torch-rb](https://github.com/ankane/torch.rb) documentation for platform-specific details.

## Setting up locally

1. **Fork and clone** the repository:

   ```bash
   git clone https://github.com/<your-username>/trainers-rb.git
   cd trainers-rb
   ```

2. **Install LibTorch** and configure Bundler (see above).

3. **Install Ruby dependencies**:

   ```bash
   bundle install
   ```

   This reads the `Gemfile` (which loads `trainers-rb.gemspec`) and installs runtime and development gems. `Gemfile.lock` is gitignored, so each contributor generates their own lockfile locally.

4. **Verify the example script** (optional, requires network access to download model weights from HuggingFace):

   ```bash
   ruby examples/finetune_sentiment.rb
   # LoRA variant:
   LORA=1 ruby examples/finetune_sentiment.rb
   ```

   Training output is written to `./output/` (gitignored).

### Environment and configuration

- No `.env` file or environment variables are required by the library itself.
- The example script accepts `LORA=1` to enable LoRA fine-tuning.
- Device selection is automatic (CPU or MPS on Apple Silicon). Use `TrainingArguments` options `no_mps: true` or `device:` to override (see README).
- First runs download pre-trained weights from HuggingFace Hub via `transformers-rb`.

## Repository structure

```
trainers-rb/
├── lib/
│   ├── trainers-rb.rb          # Gem entry point (require "trainers-rb")
│   ├── trainers.rb             # Main module, loads all components
│   └── trainers/
│       ├── version.rb          # Gem version constant
│       ├── trainer.rb          # Core training loop (train/evaluate/predict)
│       ├── trainer_utils.rb    # TrainerState, TrainerControl, helpers
│       ├── training_arguments.rb
│       ├── callbacks.rb        # Callback system + built-in callbacks
│       ├── save_utils.rb       # Model serialization via safetensors
│       ├── data/
│       │   ├── dataset.rb      # Array-of-hashes dataset wrapper
│       │   └── data_collator.rb
│       ├── optimization/
│       │   ├── optimizer.rb    # AdamW with weight-decay param groups
│       │   └── scheduler.rb    # Linear, cosine, constant LR schedules
│       └── lora/
│           ├── lora_config.rb
│           ├── lora_linear.rb
│           ├── lora_model.rb   # Apply, save, load, merge adapters
│           └── lora_utils.rb
├── examples/
│   └── finetune_sentiment.rb   # End-to-end DistilBERT fine-tuning demo
├── test/                       # Expected location for Minitest files (see below)
├── Gemfile                     # Bundler dependencies (loads gemspec)
├── trainers-rb.gemspec         # Gem specification and dependency versions
├── Rakefile                    # Test task definition
├── README.md                   # User-facing documentation
├── CHANGELOG.md                # Release history
└── LICENSE.txt                 # MIT license
```

All application code lives under `lib/`. The `Trainers` module is the public API namespace.

## Running the test suite

The project uses **Minitest** via Rake. The `Rakefile` defines:

```ruby
Rake::TestTask.new(:test) do |t|
  t.libs << "test"
  t.libs << "lib"
  t.test_files = FileList["test/**/*_test.rb"]
end

task default: :test
```

Run tests with:

```bash
bundle exec rake test
# or equivalently:
bundle exec rake
```

Place test files in `test/` using the `*_test.rb` naming convention (for example, `test/trainer_test.rb`).

> **Note:** The repository currently has no `test/` directory or test files. The Rake task is configured and ready — adding tests there is expected for new contributions that change behavior.

## Linting and static checks

The repository does **not** currently configure RuboCop, StandardRB, or other lint/format tools. There is no CI workflow checked in (no `.github/workflows/` directory).

When contributing, match the style of the existing code (see [Code style](#code-style-and-conventions) below). If you use a formatter locally, ensure it does not introduce unrelated style changes across the codebase.

## Making changes

### Branch naming

No branch naming convention is enforced in this repository. A practical recommendation:

```
<type>/<short-description>
```

Examples: `fix/gradient-accumulation-off-by-one`, `feat/streaming-dataset`, `docs/lora-save-example`.

### Keep pull requests focused

- One logical change per PR (bug fix, feature, or docs update).
- Avoid mixing unrelated refactors with functional changes.
- Update `README.md` and/or `CHANGELOG.md` when your change affects user-facing behavior or adds notable features.

### Tests

- Add or update Minitest files under `test/` for code changes that affect behavior.
- Run `bundle exec rake test` before opening a PR.
- For training-related changes, include tests that avoid downloading large models when possible (see below).

### Documentation

- Update `README.md` for new public API, changed defaults, or new usage patterns.
- Add an entry to `CHANGELOG.md` under an `[Unreleased]` section (or the next version) for user-visible changes.

## Testing ML / training functionality

Tests and examples in this project depend on the LibTorch + torch-rb stack. Keep these considerations in mind:

- **LibTorch must be installed and configured** before `bundle install` can compile `torch-rb`.
- **Model downloads**: calling `Trainers.from_pretrained` or running the example script fetches weights from HuggingFace Hub. Tests that load real models need network access and can be slow.
- **Device differences**: the library supports CPU and MPS (Apple Silicon). Behavior may differ across devices; consider testing on CPU for reproducibility (`TrainingArguments.new(no_mps: true)` or explicit `device: Torch.device("cpu")`).
- **Memory**: even small models like `distilbert-base-uncased` consume significant RAM. Use small batch sizes and short datasets in tests.
- **Determinism**: set `seed` in `TrainingArguments` when testing training behavior.
- **Output artifacts**: training writes to `output_dir` (default `./output`). The `.gitignore` excludes `output/` — do not commit generated checkpoints.

The `examples/finetune_sentiment.rb` script is a useful manual smoke test for the full training pipeline, including optional LoRA (`LORA=1`).

## Bug reports

Open a [GitHub Issue](https://github.com/trainers-rb/trainers-rb/issues) and include as much of the following as possible:

| Field | Why it helps |
|-------|--------------|
| **Ruby version** | `ruby --version` |
| **OS and architecture** | e.g. macOS arm64, Ubuntu 22.04 x86_64 |
| **LibTorch version** | The build you installed (README references 2.4.0) |
| **torch-rb version** | `bundle exec gem list torch-rb` |
| **transformers-rb version** | `bundle exec gem list transformers-rb` |
| **trainers-rb version** | Gem version or git commit SHA |
| **Steps to reproduce** | Minimal script or code snippet |
| **Expected vs. actual behavior** | What you expected and what happened |
| **Error output / logs** | Full stack trace or training log |

Security vulnerabilities should be reported privately to the maintainers rather than in a public issue.

## Feature requests

Feature requests are welcome via [GitHub Issues](https://github.com/trainers-rb/trainers-rb/issues). Please describe:

- The problem you are trying to solve.
- How you would expect the API to look or behave.
- Any relevant prior art (HuggingFace Transformers, other Ruby gems, etc.).

The README [Roadmap](https://github.com/trainers-rb/trainers-rb#roadmap) lists planned directions (mixed precision, gradient checkpointing, distributed training, QLoRA, etc.) — check there before proposing overlapping work.

## Pull request guidelines

1. Fork the repository and create a branch from `main`.
2. Make your changes with tests and documentation updates as appropriate.
3. Run `bundle exec rake test` locally.
4. Optionally run `ruby examples/finetune_sentiment.rb` to smoke-test training changes.
5. Open a pull request against `main` on [trainers-rb/trainers-rb](https://github.com/trainers-rb/trainers-rb).
6. Describe what changed, why, and how you tested it.
7. Link any related issues.

There is no automated CI in the repository yet, so clear test instructions in the PR description are especially helpful for reviewers.

## Commit messages

The repository does not define a formal commit message convention. The initial commit uses a descriptive single-line message:

```
Add trainers-rb: training loop, LoRA, and optimization utilities for fine-tuning HuggingFace transformers in Ruby
```

**Recommendation** (not an enforced rule): write a clear, imperative subject line that summarizes the change. Add a body if the *why* is not obvious. Examples:

```
Fix gradient accumulation step counting at epoch boundaries

Add EarlyStoppingCallback support for custom metric names
```

## Code style and conventions

Follow patterns already present in `lib/`:

- **`# frozen_string_literal: true`** at the top of every Ruby file.
- **Module namespace**: all public classes live under `Trainers` (e.g. `Trainers::Trainer`, `Trainers::LoraConfig`).
- **Entry point**: `require "trainers-rb"` loads the gem via `lib/trainers-rb.rb`.
- **Internal requires**: use `require_relative` for files within the gem.
- **Configuration objects**: keyword arguments in `initialize`, often backed by a frozen `DEFAULTS` hash (see `TrainingArguments`, `LoraConfig`).
- **Naming**: `snake_case` for methods and variables, `PascalCase` for classes and modules, `SCREAMING_SNAKE_CASE` for constants.
- **Indentation**: 2 spaces.
- **Comments**: brief comments for non-obvious logic (e.g. optimizer weight-decay grouping); avoid restating what the code already says.
- **Callbacks**: subclass `Trainers::TrainerCallback` and override hook methods; accept `**kwargs` for forward compatibility.

## Questions and discussion

There are no issue templates, discussion categories, or chat channels configured in this repository.

- **Bug reports and feature requests**: [GitHub Issues](https://github.com/trainers-rb/trainers-rb/issues)
- **Pull requests**: [GitHub Pull Requests](https://github.com/trainers-rb/trainers-rb/pulls)
- **Usage questions**: open an issue with the "question" label if available, or describe your question in a new issue so others can find it later.

For torch-rb or transformers-rb specific questions, refer to their respective repositories.

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE.txt).
