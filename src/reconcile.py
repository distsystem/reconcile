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


# ---------------------------------------------------------------------------
# Descriptor
# ---------------------------------------------------------------------------


# Inherit property so Pydantic treats us as a descriptor rather than
# replacing the attribute with ModelPrivateAttr during model creation.
class Dependency(property):
    fn: Callable[..., Any]
    field_name: str | None
    required: bool

    def __init__(self, fn: Callable[..., Any], *, sentinel: Any = None) -> None:
        self.fn = fn
        self._sentinel = sentinel

    def __set_name__(self, owner: type, name: str) -> None:
        if self._sentinel is None:
            # Validator: no field binding
            self.field_name = None
            self.required = False
            return

        ann = dict(owner.__annotations__) if hasattr(owner, "__annotations__") else {}

        field_name = None
        for fname in ann:
            if owner.__dict__.get(fname) is self._sentinel:
                field_name = fname
                break
        if field_name is None:
            raise TypeError(
                f"{owner.__name__}.{name}: no field found matching sentinel"
            )
        self.field_name = field_name

        if not isinstance(self._sentinel, FieldInfo):
            raise TypeError(
                f"{owner.__name__}.{field_name}: sentinel must be a pydantic "
                f"FieldInfo (use Field())"
            )

        self.required = self._sentinel.default is PydanticUndefined
        if self.required:
            # Field() has no default → Pydantic would require it at construction.
            # Inject default=None via Annotated so reconciliation can fill it later.
            # This must happen in __set_name__ because complete_model_class()
            # reads Annotated metadata AFTER __set_name__, while the metaclass
            # captures namespace defaults BEFORE it.
            self._sentinel.default = None
            ann[field_name] = typing.Annotated[ann[field_name], self._sentinel, self]
            setattr(owner, field_name, None)
            owner.__annotations__ = ann


# ---------------------------------------------------------------------------
# Decorator
# ---------------------------------------------------------------------------


def dependency(arg: Any = None, /) -> Any:
    if callable(arg) and not isinstance(arg, FieldInfo):
        return Dependency(arg)
    sentinel = arg

    def decorator(fn: Callable[..., Any]) -> Any:
        return Dependency(fn, sentinel=sentinel)

    return decorator


# ---------------------------------------------------------------------------
# Resolver
# ---------------------------------------------------------------------------


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
    return inspect.getmembers(cls, lambda a: isinstance(a, Dependency))


def reconcile[*Ts](*participants: *Ts) -> tuple[*Ts]:
    pool: dict[type, Any] = {}
    for obj in participants:
        t = type(obj)
        if t in pool:
            raise TypeError(f"Duplicate type {t.__name__} in participants")
        pool[t] = obj

    # Phase 1: Resolve — compute derived field values until convergence
    while True:
        progress = False
        for cls in [type(o) for o in pool.values() if isinstance(o, BaseModel)]:
            for _name, meta in _get_dependencies(cls):
                if meta.field_name is None or meta.field_name in pool[cls].model_fields_set:
                    continue
                method = meta.fn.__get__(pool[cls], cls)
                try:
                    result = _call(method, pool)
                except Unresolvable:
                    continue
                if result is not None:
                    setattr(pool[cls], meta.field_name, result)
                    progress = True

        if not progress:
            break

    # Phase 2: Cross-validate — run dependency validators across objects
    for obj in pool.values():
        if not isinstance(obj, BaseModel):
            continue
        cls = type(obj)
        for name, meta in _get_dependencies(cls):
            if meta.field_name is not None:
                continue
            method = meta.fn.__get__(pool[cls], cls)
            try:
                _call(method, pool)
            except Unresolvable:
                continue

    # Phase 3: Field validate — check completeness and Field constraints
    for obj in pool.values():
        if not isinstance(obj, BaseModel):
            continue
        cls = type(obj)
        for _name, meta in _get_dependencies(cls):
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
