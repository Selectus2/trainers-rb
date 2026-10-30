# Each appraisal pins torch-rb to a LibTorch generation. Regenerate the
# gemfiles with `bundle exec appraisal generate` after editing this file.

# Oldest supported: torch-rb 0.17.x builds against LibTorch 2.4.0
appraise "torch-0-17" do
  gem "torch-rb", "~> 0.17.1"
  # torch-rb 0.17 only requires rice >= 4.1; newer rice breaks its build
  gem "rice", "~> 4.3.0"
end

# Newest: torch-rb 0.27.x builds against LibTorch 2.15.0
appraise "torch-0-27" do
  gem "torch-rb", "~> 0.27.0"
end
