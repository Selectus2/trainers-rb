require_relative "test_helper"

# Issue #29: rows longer than max_length are truncated, in both input paths
class DataCollatorTest < Minitest::Test
  PadTokenizer = Struct.new(:pad_token_id)

  def collate(rows, **options)
    Trainers::DataCollatorWithPadding.new(tokenizer: PadTokenizer.new(9), **options).call(rows)
  end

  def test_arrays_are_padded_to_the_longest_row
    batch = collate([{ input_ids: [1, 2, 3], attention_mask: [1, 1, 1] }, { input_ids: [4], attention_mask: [1] }])
    assert_equal [[1, 2, 3], [4, 9, 9]], batch[:input_ids].to_a
    assert_equal [[1, 1, 1], [1, 0, 0]], batch[:attention_mask].to_a
  end

  def test_arrays_longer_than_max_length_are_truncated
    batch = collate([{ input_ids: [1, 2, 3, 4, 5] }, { input_ids: [1, 2, 3] }, { input_ids: [1] }], max_length: 3)
    assert_equal [[1, 2, 3], [1, 2, 3], [1, 9, 9]], batch[:input_ids].to_a
  end

  def test_tensors_and_arrays_give_the_same_batch
    rows = [{ input_ids: [1, 2, 3, 4, 5], labels: [0, 1, 0, 1, 0] }, { input_ids: [7], labels: [1] }]
    tensor_rows = rows.map { |row| row.transform_values { |v| Torch.tensor(v) } }

    arrays  = collate(rows, max_length: 3)
    tensors = collate(tensor_rows, max_length: 3)
    assert_equal arrays[:input_ids].to_a, tensors[:input_ids].to_a
    assert_equal arrays[:labels].to_a, tensors[:labels].to_a
  end

  def test_sequence_labels_pad_with_minus_100
    rows = [{ input_ids: [1, 2], labels: [0, 1] }, { input_ids: [3], labels: [1] }]
    assert_equal [[0, 1], [1, -100]], collate(rows)[:labels].to_a
    assert_equal [[0, 1], [1, -100]], collate(rows.map { |r| r.transform_values { |v| Torch.tensor(v) } })[:labels].to_a
  end

  def test_scalar_labels_are_stacked
    batch = collate([{ input_ids: [1], labels: 0 }, { input_ids: [2, 3], labels: 1 }])
    assert_equal [0, 1], batch[:labels].to_a
  end
end
