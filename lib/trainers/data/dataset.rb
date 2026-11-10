# frozen_string_literal: true

require "json"

module Trainers
  # Rows for training and evaluation: an Array of Hashes with Symbol keys,
  # the shape the collators and Trainer expect.
  #
  #   ds = Trainers::Dataset.from_jsonl("train.jsonl")
  #   train, eval = ds.split(test_size: 0.1)
  #   train = train.map { |row| row.merge(labels: Integer(row[:label])) }
  class Dataset
    include Enumerable

    attr_reader :data

    # One JSON object per line. Blank lines are skipped.
    def self.from_jsonl(path)
      rows = []
      File.foreach(path).with_index(1) do |line, number|
        next if line.strip.empty?

        row = begin
          JSON.parse(line, symbolize_names: true)
        rescue JSON::ParserError => e
          raise ConfigurationError, "#{path}:#{number} is not valid JSON: #{e.message}"
        end
        raise ConfigurationError, "#{path}:#{number} is a JSON #{row.class}, not an object" unless row.is_a?(Hash)

        rows << row
      end
      new(rows)
    end

    # A CSV file whose header row gives the keys. Values are Strings unless
    # csv_options say otherwise, e.g. converters: :numeric. Other options
    # (col_sep:, encoding: ...) go to CSV.
    def self.from_csv(path, **csv_options)
      require "csv"

      table = CSV.read(path, headers: true, header_converters: :symbol, **csv_options)
      new(table.map(&:to_h))
    end

    def initialize(data)
      @data = data
    end

    def [](index)
      @data[index]
    end

    def size
      @data.size
    end
    alias_method :length, :size

    def each(&block)
      @data.each(&block)
    end

    # A new Dataset when the block returns a Hash for every row. Any other
    # result gives an Array, as Enumerable#map does. The block gets a copy of
    # each row, so row.delete(:label) leaves this Dataset unchanged.
    def map(&block)
      return enum_for(:map) unless block

      results = @data.map { |row| block.call(row.is_a?(Hash) ? row.dup : row) }
      results.all?(Hash) ? self.class.new(results) : results
    end

    # A new Dataset with the rows the block returns true for. select still
    # returns an Array, as Enumerable#select does.
    def filter(&block)
      return enum_for(:filter) unless block

      self.class.new(@data.select(&block))
    end

    TOKENIZE_BATCH = 1000

    # A new Dataset with input_ids and attention_mask (and token_type_ids
    # when the tokenizer gives them) for the text in column, truncated to
    # max_length tokens. Pass two columns for text pairs, e.g. NLI. The text
    # columns are dropped unless keep: true. Takes a Tokenizers::Tokenizer or
    # a transformers-rb tokenizer.
    def tokenize(tokenizer, column: :text, max_length: 512, keep: false)
      columns = Array(column).map(&:to_sym)
      unless columns.size.between?(1, 2)
        raise ArgumentError, "column must name one text column, or two for text pairs, got #{column.inspect}"
      end
      unless max_length.is_a?(Integer) && max_length.positive?
        raise ArgumentError, "max_length must be a positive Integer, got #{max_length.inspect}"
      end

      texts = @data.each_with_index.map do |row, index|
        columns.map do |name|
          text = row[name]
          raise ConfigurationError, "Row #{index} has no String in #{name.inspect}: #{text.inspect}" unless text.is_a?(String)

          text
        end
      end

      rows = []
      texts.each_slice(TOKENIZE_BATCH).with_index do |slice, n|
        encoded = encode_texts(tokenizer, slice, max_length)
        slice.each_index do |i|
          row = @data[n * TOKENIZE_BATCH + i]
          row = row.except(*columns) unless keep
          rows << row.merge(encoded[i])
        end
      end
      self.class.new(rows)
    end

    # Shuffles with Random.new(seed) and returns [train, test]. test_size is
    # a fraction in (0, 1), or a row count.
    def split(test_size:, seed: 42)
      count = case test_size
              when Float
                raise ArgumentError, "test_size must be between 0 and 1, got #{test_size}" unless test_size > 0 && test_size < 1

                (size * test_size).round
              when Integer
                test_size
              else
                raise ArgumentError, "test_size must be a Float in (0, 1) or an Integer row count, got #{test_size.inspect}"
              end
      unless count.between?(1, size - 1)
        raise ArgumentError, "test_size #{test_size.inspect} leaves #{count} of #{size} rows for testing; " \
                             "both parts need at least one row"
      end

      rows = @data.shuffle(random: Random.new(seed))
      [self.class.new(rows.drop(count)), self.class.new(rows.first(count))]
    end

    private

    # One Hash of model inputs per text (or [text, pair])
    def encode_texts(tokenizer, texts, max_length)
      if tokenizer.respond_to?(:backend_tokenizer)
        # transformers-rb tokenizer
        first, pairs = texts.transpose
        encoded = tokenizer.(first, text_pair: pairs, truncation: true, max_length: max_length)
        keys = %i[input_ids attention_mask token_type_ids].select { |key| encoded[key] }
        texts.each_index.map { |i| keys.to_h { |key| [key, encoded[key][i]] } }
      elsif tokenizer.respond_to?(:encode_batch)
        encode_with_backend(tokenizer, texts, max_length)
      else
        raise ConfigurationError, "#{tokenizer.class} is not a Tokenizers::Tokenizer or a transformers-rb tokenizer"
      end
    end

    # Tokenizers::Tokenizer keeps truncation as a setting, so set it for the
    # call and put the caller's back afterwards
    def encode_with_backend(tokenizer, texts, max_length)
      previous = tokenizer.truncation
      tokenizer.enable_truncation(max_length)
      inputs = texts.map { |text| text.size == 1 ? text.first : text }
      tokenizer.encode_batch(inputs).map do |encoding|
        row = { input_ids: encoding.ids, attention_mask: encoding.attention_mask }
        row[:token_type_ids] = encoding.type_ids if texts.first.size == 2
        row
      end
    ensure
      tokenizer.no_truncation
      if previous
        options = previous.reject { |key, _| key == "max_length" }.transform_keys(&:to_sym)
        tokenizer.enable_truncation(previous["max_length"], **options)
      end
    end
  end
end
