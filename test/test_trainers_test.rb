require_relative "test_helper"

class TestVersion < Minitest::Test
  def test_has_version
    refute_nil Trainers::VERSION
  end

  def test_gemspec_version_matches_constant
    spec = Gem::Specification.load(File.expand_path("../trainers-rb.gemspec", __dir__))
    assert_equal Trainers::VERSION, spec.version.to_s
  end
end
