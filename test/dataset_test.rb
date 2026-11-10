require_relative "test_helper"
require "transformers-rb"

# Issues #30 and #31: loading, splitting, mapping and tokenizing datasets
class DatasetTest < Minitest::Test
  VOCAB = { "[PAD]" => 0, "[UNK]" => 1, "[CLS]" => 2, "[SEP]" => 3,
            "good" => 4, "bad" => 5, "film" => 6, "very" => 7 }.freeze

  def setup
    @tmpdir = Dir.mktmpdir
  end

  def teardown
    FileUtils.rm_rf(@tmpdir)
  end

  def test_from_jsonl
    path = write("train.jsonl", %({"text": "good film", "label": 1}\n\n{"text": "bad", "label": 0}\n))
    ds = Trainers::Dataset.from_jsonl(path)
    assert_equal [{ text: "good film", label: 1 }, { text: "bad", label: 0 }], ds.to_a
  end

  def test_from_jsonl_names_the_bad_line
    path = write("bad.jsonl", %({"text": "good"}\n{"text": oops}\n))
    error = assert_raises(Trainers::ConfigurationError) { Trainers::Dataset.from_jsonl(path) }
    assert_match "bad.jsonl:2 is not valid JSON", error.message

    path = write("array.jsonl", %([1, 2]\n))
    error = assert_raises(Trainers::ConfigurationError) { Trainers::Dataset.from_jsonl(path) }
    assert_match "array.jsonl:1 is a JSON Array, not an object", error.message
  end

  def test_from_csv_keeps_strings_unless_told
    path = write("train.csv", "text,label\ngood film,1\nbad,0\n")
    assert_equal [{ text: "good film", label: "1" }, { text: "bad", label: "0" }], Trainers::Dataset.from_csv(path).to_a
    assert_equal [1, 0], Trainers::Dataset.from_csv(path, converters: :numeric).map { |row| row[:label] }

    tsv = write("train.tsv", "text\tlabel\ngood, film\t1\n")
    assert_equal [{ text: "good, film", label: "1" }], Trainers::Dataset.from_csv(tsv, col_sep: "\t").to_a
  end

  def test_split_partitions_the_rows_deterministically
    ds = numbered(10)
    train, test = ds.split(test_size: 0.3, seed: 7)
    assert_equal [7, 3], [train.size, test.size]
    assert_equal (0...10).to_a, (train.to_a + test.to_a).map { |row| row[:id] }.sort

    again_train, again_test = ds.split(test_size: 0.3, seed: 7)
    assert_equal [train.to_a, test.to_a], [again_train.to_a, again_test.to_a]
    refute_equal test.to_a, ds.split(test_size: 0.3, seed: 8).last.to_a

    assert_equal 4, ds.split(test_size: 4).last.size
    assert_equal (0...10).to_a, ds.map { |row| row[:id] }, "split must not reorder the receiver"
  end

  def test_split_rejects_sizes_that_leave_a_part_empty
    ds = numbered(10)
    [0.0, 1.0, 1.5, 0, 10, 0.01, "0.2"].each do |size|
      assert_raises(ArgumentError, size.inspect) { ds.split(test_size: size) }
    end
  end

  def test_map_returns_a_dataset_for_rows_and_an_array_otherwise
    ds = numbered(3)
    mapped = ds.map { |row| row.merge(double: row[:id] * 2) }
    assert_instance_of Trainers::Dataset, mapped
    assert_equal [0, 2, 4], mapped.map { |row| row[:double] }
    assert_equal [{ id: 0 }, { id: 1 }, { id: 2 }], ds.to_a, "map must not mutate the receiver"
  end

  def test_filter_returns_a_dataset_and_select_an_array
    ds = numbered(5)
    assert_instance_of Trainers::Dataset, ds.filter { |row| row[:id].even? }
    assert_equal [0, 2, 4], ds.filter { |row| row[:id].even? }.map { |row| row[:id] }
    assert_instance_of Array, ds.select { |row| row[:id].even? }
  end

  def test_tokenize_with_a_tokenizers_tokenizer
    ds = Trainers::Dataset.new([{ text: "very good film", labels: 1 }, { text: "bad", labels: 0 }])
    tokenized = ds.tokenize(backend_tokenizer)
    assert_equal [{ labels: 1, input_ids: [2, 7, 4, 6, 3], attention_mask: [1, 1, 1, 1, 1] },
                  { labels: 0, input_ids: [2, 5, 3], attention_mask: [1, 1, 1] }], tokenized.to_a
  end

  def test_tokenize_with_a_transformers_tokenizer
    ds = Trainers::Dataset.new([{ text: "very good film", labels: 1 }, { text: "bad", labels: 0 }])
    tokenized = ds.tokenize(transformers_tokenizer)
    assert_equal [[2, 7, 4, 6, 3], [2, 5, 3]], tokenized.map { |row| row[:input_ids] }
    assert_equal [1, 1, 1], tokenized[1][:attention_mask]
  end

  def test_tokenize_truncates_and_keeps_the_special_tokens
    ds = Trainers::Dataset.new([{ text: "very good film" }])
    [backend_tokenizer, transformers_tokenizer].each do |tokenizer|
      assert_equal [2, 7, 4, 3], ds.tokenize(tokenizer, max_length: 4)[0][:input_ids], tokenizer.class.name
      assert_equal [2, 7, 4, 6, 3], ds.tokenize(tokenizer, max_length: 5)[0][:input_ids], tokenizer.class.name
      assert_equal [2, 7, 4, 6, 3], ds.tokenize(tokenizer, max_length: 9)[0][:input_ids], tokenizer.class.name
    end
  end

  def test_tokenize_text_pairs
    ds = Trainers::Dataset.new([{ premise: "very good", hypothesis: "film" }])
    row = ds.tokenize(backend_tokenizer, column: %i[premise hypothesis])[0]
    assert_equal [2, 7, 4, 3, 6, 3], row[:input_ids]
    assert_equal [0, 0, 0, 0, 1, 1], row[:token_type_ids]
    refute row.key?(:premise)

    assert_equal [2, 7, 4, 3, 6, 3],
                 ds.tokenize(transformers_tokenizer, column: %i[premise hypothesis])[0][:input_ids]
  end

  def test_tokenize_keeps_text_on_request_and_restores_truncation
    tokenizer = backend_tokenizer
    tokenizer.enable_truncation(3, stride: 1)
    row = Trainers::Dataset.new([{ text: "very good film" }]).tokenize(tokenizer, max_length: 4, keep: true)[0]

    assert_equal "very good film", row[:text]
    assert_equal({ "max_length" => 3, "stride" => 1 }, tokenizer.truncation.slice("max_length", "stride"))

    plain = backend_tokenizer
    Trainers::Dataset.new([{ text: "good" }]).tokenize(plain)
    assert_nil plain.truncation
  end

  def test_tokenize_errors
    ds = Trainers::Dataset.new([{ text: "good" }, { body: "bad" }])
    error = assert_raises(Trainers::ConfigurationError) { ds.tokenize(backend_tokenizer) }
    assert_match "Row 1 has no String in :text", error.message
    assert_raises(ArgumentError) { ds.tokenize(backend_tokenizer, column: %i[a b c]) }
    assert_raises(ArgumentError) { ds.tokenize(backend_tokenizer, max_length: 0) }
    assert_raises(Trainers::ConfigurationError) { ds.tokenize(Object.new) }
  end

  # From a file to a trained model with no hand-written loops
  def test_jsonl_to_training
    lines = Array.new(12) { |i| JSON.generate(text: i.even? ? "very good film" : "bad film", label: i.even? ? 1 : 0) }
    tokenizer = transformers_tokenizer
    ds = Trainers::Dataset.from_jsonl(write("reviews.jsonl", lines.join("\n")))
                          .map { |row| row.merge(labels: row.delete(:label)) }
                          .tokenize(tokenizer)
    train, eval = ds.split(test_size: 0.25)

    Trainers.set_seed(0)
    config = Transformers::Distilbert::DistilBertConfig.new(
      vocab_size: VOCAB.size, dim: 16, n_layers: 1, n_heads: 2, hidden_dim: 32, max_position_embeddings: 32, num_labels: 2
    )
    trainer = Trainers::Trainer.new(
      model: Transformers::Distilbert::DistilBertForSequenceClassification.new(config),
      args: training_args(num_train_epochs: 3, per_device_train_batch_size: 4, learning_rate: 1e-2),
      train_dataset: train, eval_dataset: eval, tokenizer: tokenizer,
      data_collator: Trainers::DataCollatorWithPadding.new(tokenizer: tokenizer)
    )
    before = quietly { trainer.evaluate }[:eval_loss]
    quietly { trainer.train }
    assert_operator quietly { trainer.evaluate }[:eval_loss], :<, before
  end

  private

  def numbered(size)
    Trainers::Dataset.new(Array.new(size) { |i| { id: i } })
  end

  def write(name, content)
    path = File.join(@tmpdir, name)
    File.write(path, content)
    path
  end

  def backend_tokenizer
    backend = Tokenizers::Tokenizer.new(Tokenizers::Models::WordLevel.new(vocab: VOCAB, unk_token: "[UNK]"))
    backend.pre_tokenizer = Tokenizers::PreTokenizers::Whitespace.new
    backend.post_processor = Tokenizers::Processors::TemplateProcessing.new(
      single: "[CLS] $A [SEP]", pair: "[CLS] $A [SEP] $B:1 [SEP]:1",
      special_tokens: [["[CLS]", VOCAB["[CLS]"]], ["[SEP]", VOCAB["[SEP]"]]]
    )
    backend
  end

  def transformers_tokenizer
    path = File.join(@tmpdir, "tokenizer.json")
    backend_tokenizer.save(path)
    Transformers::Distilbert::DistilBertTokenizerFast.new(tokenizer_file: path)
  end
end
