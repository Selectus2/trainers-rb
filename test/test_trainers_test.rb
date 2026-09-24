require_relative "test_helper"

class TestVersion < Minitest::Test
  def test_has_version
    refute_nil Trainers::VERSION
  end
end
