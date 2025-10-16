from google.protobuf import empty_pb2 as _empty_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

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

class PkCentroids(_message.Message):
    __slots__ = ("pks", "labels", "centroids")
    class Centroid(_message.Message):
        __slots__ = ("feature",)
        FEATURE_FIELD_NUMBER: _ClassVar[int]
        feature: _containers.RepeatedScalarFieldContainer[float]
        def __init__(self, feature: _Optional[_Iterable[float]] = ...) -> None: ...
    PKS_FIELD_NUMBER: _ClassVar[int]
    LABELS_FIELD_NUMBER: _ClassVar[int]
    CENTROIDS_FIELD_NUMBER: _ClassVar[int]
    pks: _containers.RepeatedScalarFieldContainer[str]
    labels: _containers.RepeatedScalarFieldContainer[int]
    centroids: _containers.RepeatedCompositeFieldContainer[PkCentroids.Centroid]
    def __init__(self, pks: _Optional[_Iterable[str]] = ..., labels: _Optional[_Iterable[int]] = ..., centroids: _Optional[_Iterable[_Union[PkCentroids.Centroid, _Mapping]]] = ...) -> None: ...
