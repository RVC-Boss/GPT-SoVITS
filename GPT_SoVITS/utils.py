"""Metadata type required to deserialize existing SoVITS checkpoints.

Released and fine-tuned checkpoints may pickle their configuration as
``utils.HParams``. Keep that type at its original module path for loading.
"""


class HParams:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if type(value) is dict:
                value = HParams(**value)
            self[key] = value

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __contains__(self, key):
        return key in self.__dict__

    def __repr__(self):
        return repr(self.__dict__)
