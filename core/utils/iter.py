import itertools
from threading import Lock

# https://www.kingname.info/2019/11/06/thread-safe-in-tee/


class SafeTee:
    def __init__(self, tee_obj, lock):
        self.tee_obj = tee_obj
        self.lock = lock

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return next(self.tee_obj)

    def __copy__(self):
        return SafeTee(self.tee_obj.__copy__(), self.lock)


def safe_tee(iterable, n=2):
    """tuple of n independent thread-safe iterators"""
    lock = Lock()
    return tuple(SafeTee(tee_obj, lock) for tee_obj in itertools.tee(iterable, n))
