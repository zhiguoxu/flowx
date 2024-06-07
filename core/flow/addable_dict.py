from __future__ import annotations


class AddableDict(dict):
    def __add__(self, other: dict) -> AddableDict:
        if not isinstance(other, dict):
            return NotImplemented

        # copy self
        result = AddableDict(self)

        for key in other:
            if result.get(key) is None:
                result[key] = other[key]
            elif other[key] is not None:
                try:
                    result[key] += other[key]
                except TypeError:
                    result[key] = other[key]
        return result

    def __radd__(self, other: dict) -> AddableDict:
        return AddableDict(other) + self
