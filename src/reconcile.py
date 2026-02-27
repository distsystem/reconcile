"""reconcile — declarative cross-object field resolution for Pydantic models.

Single entry point ``reconcile`` serves as decorator (field derivation /
validator) and resolver, dispatched by the first argument.
"""

import inspect
import typing
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel
from pydantic.fields import FieldInfo


# ---------------------------------------------------------------------------
# Descriptor
# ---------------------------------------------------------------------------


# Inherit property so Pydantic treats us as a descriptor rather than
# replacing the attribute with ModelPrivateAttr during model creation.
class Reconciled(property):
    UNRESOLVED: Any = object()

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
                f"FieldInfo (use Field(default=Reconciled.UNRESOLVED))"
            )

        self.required = self._sentinel.default is Reconciled.UNRESOLVED
        # Replace sentinel with valid default so Pydantic validation accepts it
        if self.required:
            self._sentinel.default = None
        # Move FieldInfo from class-body value into Annotated metadata,
        # where Pydantic's model metaclass picks it up
        ann[field_name] = typing.Annotated[ann[field_name], self._sentinel, self]
        # Replace descriptor on the class with a plain default so
        # BaseModel.__init__ sees a normal attribute, not our sentinel
        setattr(owner, field_name, None)

        owner.__annotations__ = ann


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------


def _inject_from_pool(
    fn: Callable[..., Any], pool: dict[type, Any]
) -> tuple[bool, dict[str, Any]]:
    hints = typing.get_type_hints(fn)
    kwargs: dict[str, Any] = {}
    for param in inspect.signature(fn).parameters.values():
        annotation = hints.get(param.name, param.annotation)
        if annotation is inspect.Parameter.empty or annotation not in pool:
            return False, {}
        kwargs[param.name] = pool[annotation]
    return True, kwargs


def _find_reconciled(cls: type) -> list[tuple[str, Reconciled]]:
    return [
        (name, attr)
        for name in dir(cls)
        if isinstance(attr := getattr(cls, name, None), Reconciled)
    ]


def _resolve(*participants: Any, max_iterations: int = 10) -> tuple[Any, ...]:
    pool: dict[type, Any] = {}
    for obj in participants:
        t = type(obj)
        if t in pool:
            raise TypeError(f"Duplicate type {t.__name__} in participants")
        pool[t] = obj

    errors: list[str] = []
    ran: set[tuple[type, str]] = set()

    for _ in range(max_iterations):
        progress = False
        for obj in list(pool.values()):
            if not isinstance(obj, BaseModel):
                continue
            cls = type(obj)
            updates: dict[str, Any] = {}

            for name, meta in _find_reconciled(cls):
                field = meta.field_name
                if field is not None:
                    if field in obj.model_fields_set:
                        continue
                else:
                    if (cls, name) in ran:
                        continue

                method = meta.fn.__get__(pool[cls], cls)
                resolvable, kwargs = _inject_from_pool(method, pool)
                if not resolvable:
                    continue
                try:
                    result = method(**kwargs)
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
        for _name, meta in _find_reconciled(cls):
            if not meta.required or meta.field_name is None:
                continue
            if meta.field_name in obj.model_fields_set:
                continue
            method = meta.fn.__get__(obj, cls)
            resolvable, _ = _inject_from_pool(method, pool)
            if not resolvable:
                errors.append(
                    f"{cls.__name__}.{meta.field_name}: required but unresolved"
                )

    if errors:
        raise ValueError(
            "Constraint violations:\n" + "\n".join(f"  - {msg}" for msg in errors)
        )

    return tuple(pool[type(p)] for p in participants)


def reconcile(*args: Any, **kwargs: Any) -> Any:
    if not args:
        raise TypeError("reconcile() requires at least one argument")
    first = args[0]
    # Multiple args, BaseModel instance, or kwargs → resolver
    if len(args) > 1 or isinstance(first, BaseModel) or kwargs:
        return _resolve(*args, **kwargs)
    # Callable (but not FieldInfo) → bare @reconcile decorator (validator)
    if callable(first) and not isinstance(first, FieldInfo):
        return Reconciled(first)
    # Sentinel → @reconcile(sentinel) decorator factory (field derivation)
    sentinel = first

    def decorator(fn: Callable[..., Any]) -> Any:
        return Reconciled(fn, sentinel=sentinel)

    return decorator
