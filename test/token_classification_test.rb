require_relative "test_helper"

# A per-token classifier: logits of shape [batch, seq, labels]
class TinyTagger < Torch::NN::Module
  def initialize(vocab = 10, num_labels = 3)
    super()
    @embed = Torch::NN::Embedding.new(vocab, num_labels)
  end

  def forward(input_ids:, attention_mask: nil, labels: nil)
    @embed.call(input_ids)
  end
end

# Returns its own loss, like transformers-rb's BertForTokenClassification
class TinyTaggerWithLoss < TinyTagger
  Output = Struct.new(:loss, :logits, keyword_init: true)

  def forward(input_ids:, attention_mask: nil, labels: nil)
    logits = super
    loss   = labels && Torch::NN::F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
    Output.new(loss: loss, logits: logits)
  end
end

# Issue #27: token classification trains, evaluates and predicts, with
# batches padded to different lengths
class TokenClassificationTest < Minitest::Test
  PadTokenizer = Struct.new(:pad_token_id)

  def setup
    @tmpdir = Dir.mktmpdir
  end

  def teardown
    FileUtils.rm_rf(@tmpdir)
  end

  # Rows of 1, 2, 3, 4 and 5 tokens; batch 2 gives batches of length 2, 4 and 5
  def dataset
    rows = Array.new(5) do |i|
      ids = Array.new(i + 1) { |j| (i + j) % 10 }
      { input_ids: ids, attention_mask: [1] * ids.size, labels: ids.map { |id| id % 3 } }
    end
    Trainers::Dataset.new(rows)
  end

  def trainer(model, **options)
    Trainers::Trainer.new(
      model: model, args: training_args(num_train_epochs: 2, per_device_train_batch_size: 2,
                                        per_device_eval_batch_size: 2, learning_rate: 0.1),
      train_dataset: dataset, eval_dataset: dataset,
      data_collator: Trainers::DataCollatorWithPadding.new(tokenizer: PadTokenizer.new(0)), **options
    )
  end

  def test_trains_evaluates_and_predicts
    [TinyTagger, TinyTaggerWithLoss].each do |klass|
      Trainers.set_seed(1)
      t = trainer(klass.new)
      before = quietly { t.evaluate }[:eval_loss]
      quietly { t.train }
      after = quietly { t.evaluate }[:eval_loss]
      assert_operator after, :<, before, klass.name

      preds = t.predict(dataset)
      assert_equal [5, 5, 3], preds.shape, klass.name
    end
  end

  # transformers-rb 0.2.0's BertForTokenClassification raises NameError
  # (CrossEntropyLoss) when given labels; the Trainer computes the loss instead
  def test_transformers_bert_token_classification
    require "transformers-rb"
    Trainers.set_seed(0)
    config = Transformers::Bert::BertConfig.new(vocab_size: 10, hidden_size: 8, num_hidden_layers: 1,
                                                num_attention_heads: 2, intermediate_size: 16, num_labels: 3)
    t = trainer(Transformers::Bert::BertForTokenClassification.new(config))
    before = quietly { t.evaluate }[:eval_loss]
    quietly { t.train }

    assert_operator quietly { t.evaluate }[:eval_loss], :<, before
    assert_equal [5, 5, 3], t.predict(dataset).shape
  end

  def test_predictions_and_labels_line_up_for_compute_metrics
    received = nil
    metrics = lambda do |pred|
      received = pred
      labels = pred.label_ids
      mask   = labels.ne(-100)
      { accuracy: pred.predictions.argmax(-1).eq(labels).logical_and(mask).sum.item.to_f / mask.sum.item }
    end
    result = quietly { trainer(TinyTagger.new, compute_metrics: metrics).evaluate }

    assert_equal [5, 5, 3], received.predictions.shape
    assert_equal [5, 5], received.label_ids.shape
    # The one-token row is padded with -100 after its label
    assert_equal [0, -100, -100, -100, -100], received.label_ids[0].to_a
    assert_includes 0.0..1.0, result[:accuracy]
  end

  def test_padded_tokens_do_not_count_towards_the_loss
    model = TinyTagger.new
    rows  = [{ input_ids: [1, 2], labels: [0, 1] }, { input_ids: [3], labels: [2] }]
    batch = Trainers::DataCollatorWithPadding.new(tokenizer: PadTokenizer.new(0)).call(rows)
    logits = model.(input_ids: batch[:input_ids])
    expected = Torch::NN::F.cross_entropy(
      Torch.stack([logits[0][0], logits[0][1], logits[1][0]]), Torch.tensor([0, 1, 2])
    )

    t = trainer(model)
    loss = t.send(:loss_for, logits, logits, batch[:labels])
    assert_in_delta expected.item, loss.item, 1e-6
  end
end

# Issue #27: eval_loss weights each batch by its size
class EvalLossWeightingTest < Minitest::Test
  def test_a_short_last_batch_counts_for_its_rows_only
    model = TinyClassifier.new
    data  = synthetic_dataset(5)
    args  = training_args(per_device_eval_batch_size: 2)
    t     = Trainers::Trainer.new(model: model, args: args, train_dataset: data, eval_dataset: data)

    inputs = Torch.tensor(data.map { |row| row[:input] })
    labels = Torch.tensor(data.map { |row| row[:labels] })
    expected = Torch.no_grad { Torch::NN::F.cross_entropy(model.(input: inputs), labels) }

    assert_in_delta expected.item, quietly { t.evaluate }[:eval_loss], 1e-6
  end
end
