from functools import wraps
from contextlib import contextmanager
import time
import sys

def timer(f):
    @wraps(f)
    def timer_inner_wrapper(*args, **kwargs):
        start = time.time()
        result = f(*args, **kwargs)
        end = time.time()
        print(f" -- {f.__name__} [{end - start:.2f}]")
        return result
    return timer_inner_wrapper

timer_stack = []

# https://realpython.com/python-with-statement/#creating-custom-context-managers
class Timer:
    def __init__(self, msg=None, context=None):
        self.msg = msg
        if len(timer_stack) > 0:
            self.context = timer_stack[-1]
        else:
            self.context = None

        if self.context:
            self.level = self.context.level + 1
        else:
            self.level = 0

        self.prefix = ' ' * (self.level * 4)

    def __enter__(self):
        self.start = time.time()
        self.end = 0.0
        # print(self.context)
        print(f"{self.prefix} -- {self.msg} [start]", file=sys.stderr)
        # print(f"pushed {self.msg}")
        timer_stack.append(self)
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.elapsed = self.end - self.start
        if self.msg:
            print(f"{self.prefix} -- {self.msg} [{self.elapsed:.2f}]", file=sys.stderr)
        else:
            print(f"{self.prefix} -- [{self.elapsed:.2f}]", file=sys.stderr)
        popped = timer_stack.pop()
        # print(f"popped {popped.msg}")

if __name__ == '__main__':
    with Timer("Outer Function") as _:
        time.sleep(0.5)
        with Timer("Inner fn 1") as _:
            time.sleep(2.0)
            with Timer("quick") as _:
                time.sleep(0.001)

        with Timer("Inner fn 2") as _:
            time.sleep(1.0)
