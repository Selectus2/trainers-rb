# frozen_string_literal: true

require "logger"

module Trainers
  class << self
    # Where trainers-rb reports progress. Defaults to $stderr at INFO.
    # Set to any Logger-compatible object, or nil for a silent library.
    attr_accessor :logger

    def default_logger
      Logger.new($stderr, level: Logger::INFO, progname: "trainers").tap do |logger|
        logger.formatter = proc do |severity, _time, progname, message|
          prefix = severity == "INFO" ? "" : "#{severity}: "
          "[#{progname}] #{prefix}#{message}\n"
        end
      end
    end

    # Internal: log through Trainers.logger when one is set
    def log(level, message)
      logger&.public_send(level, message)
    end
  end

  self.logger = default_logger
end
