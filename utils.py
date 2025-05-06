# Helper function for profiling

from time import perf_counter
import sys
import os

STDERR_FILENO = 2


class catchtime:
    """
    Usage:
    with catchtime("message"):
        do_stuff()
    """

    def __init__(self, msg, logfile):
        self.msg = msg
        self.logfile = logfile

    def __enter__(self):
        self.start = perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        self.time = perf_counter() - self.start
        self.readout = f"Time[{self.msg}]: {self.time:.6f}s"
        print(self.readout, file=self.logfile)


class swap_stderr:
    # NOTE: these must be "saved" here to avoid exceptions when using
    #       this context manager inside of a __del__ method
    sys = sys
    os = os

    def __init__(self, logfile):
        self.logfile = logfile

    # Oddly enough this works better than the contextlib version
    def __enter__(self):
        self.old_stderr_fileno_undup = STDERR_FILENO
        self.old_stderr_fileno = self.os.dup(self.old_stderr_fileno_undup)
        self.old_stderr = self.sys.stderr

        self.os.dup2(self.logfile.fileno(), self.old_stderr_fileno_undup)

        self.sys.stderr = self.logfile
        return self

    def __exit__(self, *_):
        # Check if sys.stdout and sys.stderr have fileno method
        self.sys.stderr = self.old_stderr
        self.os.dup2(self.old_stderr_fileno, self.old_stderr_fileno_undup)
        self.os.close(self.old_stderr_fileno)
