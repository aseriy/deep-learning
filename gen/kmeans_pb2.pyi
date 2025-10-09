from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class PkVector(_message.Message):
    __slots__ = ("pk", "vector")
    PK_FIELD_NUMBER: _ClassVar[int]
    VECTOR_FIELD_NUMBER: _ClassVar[int]
    pk: str
    vector: _containers.RepeatedScalarFieldContainer[float]
    def __init__(self, pk: _Optional[str] = ..., vector: _Optional[_Iterable[float]] = ...) -> None: ...

class PkVectorAck(_message.Message):
    __slots__ = ("ok",)
    OK_FIELD_NUMBER: _ClassVar[int]
    ok: bool
    def __init__(self, ok: bool = ...) -> None: ...
