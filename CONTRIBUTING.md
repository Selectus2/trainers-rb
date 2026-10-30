# Contributing to trainers-rb

Thanks for helping. The one real hurdle is LibTorch, the C++ library that torch-rb compiles against. Once it is installed, everything else is ordinary Ruby.

## 1. Ruby

trainers-rb is tested on Ruby 3.3, 3.4 and 4.0. Any version manager works; with RVM:

```bash
rvm install 3.3.6
rvm use 3.3.6
```

## 2. LibTorch

Each torch-rb release builds against one LibTorch version. CI tests two pairs:

| Gemfile | torch-rb | LibTorch |
|---|---|---|
| `gemfiles/torch_0_17.gemfile` | 0.17.x | 2.4.0 |
| `gemfiles/torch_0_27.gemfile` (and the root `Gemfile`) | 0.27.x | 2.15.0 |

For day-to-day work you only need LibTorch 2.15.0. Install 2.4.0 too if you are changing anything that touches torch-rb APIs.

### macOS (Apple Silicon)

```bash
V=2.15.0
curl -L -o /tmp/libtorch.zip https://download.pytorch.org/libtorch/cpu/libtorch-macos-arm64-$V.zip
mkdir -p ~/libtorch-$V && unzip -q /tmp/libtorch.zip -d ~/libtorch-$V
brew install libomp   # LibTorch links against OpenMP
```

### Linux (x86_64)

```bash
V=2.15.0
# For 2.7 and older, the file is libtorch-cxx11-abi-shared-with-deps-$V%2Bcpu.zip
wget -O /tmp/libtorch.zip "https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-$V%2Bcpu.zip"
mkdir -p ~/libtorch-$V && unzip -q /tmp/libtorch.zip -d ~/libtorch-$V
```

Either way, LibTorch ends up in `~/libtorch-<version>/libtorch`.

## 3. Install and run the tests

Tell Bundler where LibTorch is, then install. Compiling torch-rb takes a few minutes the first time.

```bash
bundle config set --local build.torch-rb --with-torch-dir=$HOME/libtorch-2.15.0/libtorch
bundle install
bundle exec rake test
```

The suite uses tiny synthetic models, never downloads anything and runs in about a second.

### Against a specific torch-rb version

Both Appraisal gemfiles share one Bundler config, so pass the LibTorch path as an environment variable instead:

```bash
BUNDLE_GEMFILE=gemfiles/torch_0_17.gemfile \
BUNDLE_BUILD__TORCH___RB=--with-torch-dir=$HOME/libtorch-2.4.0/libtorch \
  bundle install

BUNDLE_GEMFILE=gemfiles/torch_0_17.gemfile bundle exec rake test
```

After editing `Appraisals`, regenerate the gemfiles with `bundle exec appraisal generate`.

### What CI runs

Ruby 3.3, 3.4 and 4.0 on Linux against both gemfiles, plus Ruby 4.0 with torch-rb 0.27 on macOS arm64. The torch-rb 0.17 jobs on Ruby 3.4 and 4.0 are allowed to fail. They predate those Rubies, and are kept to show when the oldest pair stops building.

## What a good PR looks like

- **One issue per PR.** Link it with `Fixes #N`.
- **Tests first for bugs.** Add a test that fails without your fix. Tests go in `test/*_test.rb`, and fixtures live in `test/test_helper.rb`.
- **No network, no pretrained weights in tests.** Use the synthetic models in the test helper.
- **Public API changes are deliberate.** `test/public_api_test.rb` fails when a public method or keyword changes. If the change is intended, update that test and add a CHANGELOG entry under "Unreleased".
- **Behaviour changes need a CHANGELOG line**, even small ones.
- **Keep the diff focused.** Don't mix refactors with fixes.

## Reporting bugs

Open an issue with your Ruby, torch-rb and LibTorch versions, plus the smallest script that reproduces the problem. For security issues, see [SECURITY.md](SECURITY.md) instead.
