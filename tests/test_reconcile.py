"""Tests — 9 test cases."""

from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict, Field

from reconcile import dependency, reconcile


# ---------------------------------------------------------------------------
# Test models
# ---------------------------------------------------------------------------


class TrainingSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    num_steps: int = 1000


class CrossEntropyLoss(BaseModel):
    ignore_index: int = -100
    compile: bool = False

    def __call__(self, _logits: Any, _labels: Any) -> str:
        return f"ce_loss(ignore_index={self.ignore_index})"


class AdamWOptimizerSpec(BaseModel):
    lr: float = 1e-3
    betas: tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 0.01

    @dependency
    def _lr_positive(self, _t: TrainingSpec) -> None:
        if self.lr <= 0:
            raise ValueError(f"lr={self.lr} must be positive")


class LinearWarmupSchedulerSpec(BaseModel):
    warmup_steps: int = 0
    lr_min: float = 0.0
    num_steps: int = Field()

    @dependency(num_steps)
    def _(self, t: TrainingSpec) -> int:
        if self.warmup_steps >= t.num_steps:
            raise ValueError(f"warmup ({self.warmup_steps}) >= total ({t.num_steps})")
        return t.num_steps

    def func(self, step: int) -> float:
        return step * self.lr_min


class MultiFieldSpec(BaseModel):
    num_steps: int = Field()
    lr: float = Field()

    @dependency(num_steps)
    def _derive_num_steps(self, t: TrainingSpec) -> int:
        return t.num_steps

    @dependency(lr)
    def _derive_lr(self, o: AdamWOptimizerSpec) -> float:
        return o.lr


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_cross_object_resolution():
    """1. Cross-object dependency resolution."""
    sched, _training = reconcile(
        LinearWarmupSchedulerSpec(warmup_steps=100),
        TrainingSpec(num_steps=2000),
    )
    assert sched.num_steps == 2000


def test_manual_override():
    """2. Manual override wins over reconcile."""
    sched, _ = reconcile(
        LinearWarmupSchedulerSpec(warmup_steps=100, num_steps=999),
        TrainingSpec(num_steps=2000),
    )
    assert sched.num_steps == 999

    # Partial override on multi-field model
    multi, _, _ = reconcile(
        MultiFieldSpec(lr=1e-2),
        TrainingSpec(num_steps=3000),
        AdamWOptimizerSpec(lr=5e-4),
    )
    assert multi.lr == 1e-2
    assert multi.num_steps == 3000


def test_multi_participant():
    """3. Realization is user code, not framework protocol."""
    loss, optim, sched, _training, multi = reconcile(
        CrossEntropyLoss(),
        AdamWOptimizerSpec(lr=3e-4),
        LinearWarmupSchedulerSpec(warmup_steps=200),
        TrainingSpec(num_steps=5000),
        MultiFieldSpec(),
    )
    assert loss("logits", "labels") == "ce_loss(ignore_index=-100)"
    assert optim.lr == 3e-4
    assert sched.num_steps == 5000
    assert multi.num_steps == 5000
    assert multi.lr == 3e-4


def test_duplicate_type():
    """4. Error: duplicate type."""
    with pytest.raises(TypeError, match="Duplicate type"):
        reconcile(TrainingSpec(), TrainingSpec())


def test_required_unresolved():
    """5. Error: required field unresolved."""
    with pytest.raises(ValueError, match="required but unresolved"):
        reconcile(LinearWarmupSchedulerSpec(warmup_steps=100))
    with pytest.raises(ValueError, match="required but unresolved"):
        reconcile(MultiFieldSpec())


def test_derivation_validation_error():
    """6. Error: derivation raises on validation failure."""
    with pytest.raises(ValueError, match=r"warmup \(5000\) >= total \(2000\)"):
        reconcile(
            LinearWarmupSchedulerSpec(warmup_steps=5000),
            TrainingSpec(num_steps=2000),
        )


def test_skip_when_dependency_absent():
    """7. Graceful skip when dependency absent but field already set."""
    (sched,) = reconcile(LinearWarmupSchedulerSpec(warmup_steps=100, num_steps=500))
    assert sched.num_steps == 500


def test_fail_fast():
    """8. Validation fails fast on first error."""
    with pytest.raises(ValueError):
        reconcile(
            AdamWOptimizerSpec(lr=0),
            LinearWarmupSchedulerSpec(warmup_steps=5000),
            TrainingSpec(num_steps=2000),
        )


def test_field_constraints_validated():
    """9. Field constraints are enforced on reconciled values."""

    class Bounded(BaseModel):
        value: int = Field(ge=0, le=100)

        @dependency(value)
        def _(self, t: TrainingSpec) -> int:
            return t.num_steps

    with pytest.raises(ValueError, match="less than or equal to 100"):
        reconcile(Bounded(), TrainingSpec(num_steps=9999))

    bounded, _ = reconcile(Bounded(), TrainingSpec(num_steps=50))
    assert bounded.value == 50


def test_model_fields_and_dump():
    """10. num_steps is a real Pydantic field."""
    assert "num_steps" in LinearWarmupSchedulerSpec.model_fields
    assert LinearWarmupSchedulerSpec().model_dump() == {
        "warmup_steps": 0,
        "lr_min": 0.0,
        "num_steps": None,
    }
    assert LinearWarmupSchedulerSpec(num_steps=42).num_steps == 42
