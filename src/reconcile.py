"""reconcile — declarative cross-object field resolution for Pydantic models.

``dependency`` declares cross-object field derivations and validators.
``reconcile`` resolves all dependencies to a consistent state.
"""

import inspect
import typing
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel
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


def reconcile(*participants: Any) -> tuple[Any, ...]:
    pool: dict[type, Any] = {}
    for obj in participants:
        t = type(obj)
        if t in pool:
            raise TypeError(f"Duplicate type {t.__name__} in participants")
        pool[t] = obj

    errors: list[str] = []
    ran: set[tuple[type, str]] = set()

    while True:
        progress = False
        for obj in list(pool.values()):
            if not isinstance(obj, BaseModel):
                continue
            cls = type(obj)
            updates: dict[str, Any] = {}

            for name, meta in inspect.getmembers(
                type(obj), lambda a: isinstance(a, Dependency)
            ):
                field = meta.field_name
                if field is not None:
                    if field in obj.model_fields_set:
                        continue
                else:
                    if (cls, name) in ran:
                        continue

                method = meta.fn.__get__(pool[cls], cls)
                try:
                    result = _call(method, pool)
                except Unresolvable:
                    continue
                except ValueError as e:
                    label = field or name
                    errors.append(f"{cls.__name__}.{label}: {e}")
                    continue

                if field is not None and result is not None:
                    updates[field] = result
                else:
                    ran.add((cls, name))

            if updates:
                pool[cls] = obj.model_copy(update=updates)
                progress = True

        if not progress:
            break

    for obj in pool.values():
        if not isinstance(obj, BaseModel):
            continue
        cls = type(obj)
        for _name, meta in inspect.getmembers(
            type(obj), lambda a: isinstance(a, Dependency)
        ):
            if not meta.required or meta.field_name is None:
                continue
            if meta.field_name not in obj.model_fields_set:
                errors.append(
                    f"{cls.__name__}.{meta.field_name}: required but unresolved"
                )

    if errors:
        raise ValueError(
            "Constraint violations:\n" + "\n".join(f"  - {msg}" for msg in errors)
        )

    return tuple(pool[type(p)] for p in participants)
