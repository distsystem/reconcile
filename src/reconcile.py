"""reconcile — declarative cross-object field resolution for Pydantic models.

``dependency`` declares cross-object field derivations and validators.
``reconcile`` resolves all dependencies to a consistent state.
"""

import inspect
import typing
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, TypeAdapter
from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefined


# id(FieldInfo) → list[Dependency]: registered at decorator time, consumed
# by __set_name__ before Pydantic's complete_model_class() reads annotations.
_registry: dict[int, list[Dependency]] = {}


# Inherit property so Pydantic treats us as a descriptor rather than
# replacing the attribute with ModelPrivateAttr during model creation.
class Dependency(property):
    fn: Callable[..., Any]
    field_name: str | None
    required: bool

    def __init__(self, fn: Callable[..., Any], *, sentinel: Any = None) -> None:
        self.fn = fn
        self.field_name = None
        self.required = False
        if isinstance(sentinel, FieldInfo):
            _registry.setdefault(id(sentinel), []).append(self)

    def __set_name__(self, owner: type, name: str) -> None:
        ann = dict(owner.__annotations__)
        for fname in ann:
            fi = owner.__dict__.get(fname)
            if not isinstance(fi, FieldInfo):
                continue
            for dep in _registry.pop(id(fi), []):
                dep.field_name = fname
                dep.required = fi.default is PydanticUndefined
                if dep.required:
                    fi.default = None
                ann[fname] = typing.Annotated[ann[fname], fi, dep]
                setattr(owner, fname, fi.default)
        owner.__annotations__ = ann


def dependency(arg: Any = None, /) -> Any:
    if callable(arg) and not isinstance(arg, FieldInfo):
        return Dependency(arg)
    sentinel = arg

    def decorator(fn: Callable[..., Any]) -> Any:
        return Dependency(fn, sentinel=sentinel)

    return decorator


class Unresolvable(Exception):
    pass


class Pool:
    _EXCLUDED: typing.ClassVar[set[type]] = {object, BaseModel}

    def __init__(self, participants: tuple[Any, ...]) -> None:
        self._data: dict[type, list[Any]] = {}
        self._deps: dict[type, list[Dependency]] = {}
        for obj in participants:
            for cls in type(obj).__mro__:
                if cls in self._EXCLUDED:
                    continue
                self._data.setdefault(cls, []).append(obj)

    def resolve(self, requested: type) -> Any:
        candidates = self._data.get(requested, [])
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            names = ", ".join(type(c).__name__ for c in candidates)
            raise TypeError(
                f"Ambiguous: multiple candidates for {requested.__name__}: {names}"
            )
        raise KeyError(requested)

    def call(self, fn: Callable[..., Any]) -> Any:
        hints = typing.get_type_hints(fn)
        hints.pop("return", None)
        try:
            kwargs = {p: self.resolve(t) for p, t in hints.items()}
        except KeyError:
            raise Unresolvable
        return fn(**kwargs)

    def deps(self, cls: type) -> list[Dependency]:
        if cls not in self._deps:
            result = [d for _, d in inspect.getmembers(cls, lambda a: isinstance(a, Dependency))]
            seen = {id(d) for d in result}
            for fi in cls.model_fields.values():
                for m in fi.metadata:
                    if isinstance(m, Dependency) and id(m) not in seen:
                        result.append(m)
                        seen.add(id(m))
            self._deps[cls] = result
        return self._deps[cls]


def reconcile[*Ts](*participants: *Ts) -> tuple[*Ts]:
    pool = Pool(participants)
    models = [obj for obj in participants if isinstance(obj, BaseModel)]

    # Phase 1: Resolve — compute derived field values until convergence
    while True:
        progress = False
        for obj in models:
            cls = type(obj)
            for meta in pool.deps(cls):
                if meta.field_name is None or meta.field_name in obj.model_fields_set:
                    continue
                try:
                    result = pool.call(meta.fn.__get__(obj, cls))
                except Unresolvable:
                    continue
                if result is not None:
                    setattr(obj, meta.field_name, result)
                    progress = True
        if not progress:
            break

    # Phase 2: Cross-validate — run dependency validators across objects
    for obj in models:
        cls = type(obj)
        for meta in pool.deps(cls):
            if meta.field_name is not None:
                continue
            try:
                pool.call(meta.fn.__get__(obj, cls))
            except Unresolvable:
                continue

    # Phase 3: Field validate — check completeness and Field constraints
    for obj in models:
        cls = type(obj)
        for meta in pool.deps(cls):
            if not meta.required or meta.field_name is None:
                continue
            if meta.field_name not in obj.model_fields_set:
                raise ValueError(
                    f"{cls.__name__}.{meta.field_name}: required but unresolved"
                )
        for field_name in obj.model_fields_set:
            fi = cls.model_fields[field_name]
            if fi.metadata:
                ta = TypeAdapter(typing.Annotated[fi.annotation, *fi.metadata])
                ta.validate_python(getattr(obj, field_name))

    return participants
