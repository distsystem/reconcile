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


def _call(fn: Callable[..., Any], pool: dict[type, Any]) -> Any:
    hints = typing.get_type_hints(fn)
    hints.pop("return", None)
    try:
        kwargs = {p: pool[t] for p, t in hints.items()}
    except KeyError:
        raise Unresolvable
    return fn(**kwargs)


def _get_dependencies(cls: type) -> list[tuple[str, Dependency]]:
    deps = inspect.getmembers(cls, lambda a: isinstance(a, Dependency))
    seen = {id(d) for _, d in deps}
    for fname, fi in cls.model_fields.items():
        for m in fi.metadata:
            if isinstance(m, Dependency) and id(m) not in seen:
                deps.append((fname, m))
                seen.add(id(m))
    return deps


def reconcile[*Ts](*participants: *Ts) -> tuple[*Ts]:
    pool: dict[type, Any] = {}
    for obj in participants:
        t = type(obj)
        if t in pool:
            raise TypeError(f"Duplicate type {t.__name__} in participants")
        pool[t] = obj

    model_deps = [
        (cls, _get_dependencies(cls))
        for cls in pool
        if isinstance(pool[cls], BaseModel)
    ]

    # Phase 1: Resolve — compute derived field values until convergence
    while True:
        progress = False
        for cls, deps in model_deps:
            obj = pool[cls]
            for _, meta in deps:
                if meta.field_name is None or meta.field_name in obj.model_fields_set:
                    continue
                try:
                    result = _call(meta.fn.__get__(obj, cls), pool)
                except Unresolvable:
                    continue
                if result is not None:
                    setattr(obj, meta.field_name, result)
                    progress = True
        if not progress:
            break

    # Phase 2: Cross-validate — run dependency validators across objects
    for cls, deps in model_deps:
        obj = pool[cls]
        for _, meta in deps:
            if meta.field_name is not None:
                continue
            try:
                _call(meta.fn.__get__(obj, cls), pool)
            except Unresolvable:
                continue

    # Phase 3: Field validate — check completeness and Field constraints
    for cls, deps in model_deps:
        obj = pool[cls]
        for _, meta in deps:
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
