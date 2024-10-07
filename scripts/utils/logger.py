import sys

class Logger(object):
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()  # Ensure it writes to file in real-time

    def flush(self):
        pass

def setup_logger(log_file):
    sys.stdout = Logger(log_file)
    sys.stderr = Logger(log_file)  # Also log stderr if needed