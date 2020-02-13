"""
Logging is very important. It helps you:
- look at old experiments and see what happened
- track down bugs
- monitor ongoing experiments
and many other things.

My current favorite is loguru https://loguru.readthedocs.io/en/stable/index.html
"""

def setup(log_fname=None):
    raise NotImplementedError

def debug(s):
    raise NotImplementedError

def info(s):
    raise NotImplementedError
