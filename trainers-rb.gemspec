Gem::Specification.new do |spec|
  spec.name          = "trainers-rb"
  spec.version       = "0.1.0"
  spec.authors       = ["Vishwajeetsingh Desurkar"]
  spec.email         = ["vishwajeetsinghd@gmail.com"]
  spec.summary       = "Fine-tune transformer models in Ruby"
  spec.description   = "Training loop, LoRA, and optimization utilities for fine-tuning " \
                        "HuggingFace transformer models using torch-rb and transformers-rb. " \
                        "Supports full fine-tuning, LoRA adapters, learning rate scheduling, " \
                        "callbacks, and model serialization via safetensors."
  spec.homepage      = "https://github.com/Selectus2/trainers-rb"
  spec.license       = "MIT"
  spec.required_ruby_version = ">= 3.1.0"

  spec.metadata = {
    "homepage_uri"      => "https://github.com/Selectus2/trainers-rb",
    "source_code_uri"   => "https://github.com/Selectus2/trainers-rb",
    "changelog_uri"     => "https://github.com/Selectus2/trainers-rb/blob/main/CHANGELOG.md",
    "bug_tracker_uri"   => "https://github.com/Selectus2/trainers-rb/issues",
    "rubygems_mfa_required" => "true"
  }

  spec.files = Dir["lib/**/*.rb", "LICENSE.txt", "README.md", "CHANGELOG.md"]
  spec.require_paths = ["lib"]

  spec.add_dependency "torch-rb",        ">= 0.17.1"
  spec.add_dependency "transformers-rb", ">= 0.2.0"
  spec.add_dependency "safetensors",     ">= 0.1.1"
  spec.add_dependency "tokenizers",      ">= 0.5.3"

  spec.add_development_dependency "rake",     "~> 13.0"
  spec.add_development_dependency "minitest", "~> 5.0"
end
