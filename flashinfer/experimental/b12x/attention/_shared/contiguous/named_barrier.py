import enum


class NamedBarrierFwd(enum.IntEnum):
    Epilogue = enum.auto()
    PFull = enum.auto()
    PEmpty = enum.auto()
    KVConvert = enum.auto()
