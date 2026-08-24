"""Generic implementation of operators including basic arithmetic."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from enum import Enum
from math import prod
from numbers import Integral, Number
from typing import Any, Literal
from copy import deepcopy

from gratopy.utilities import Numeric


class OperatorArithmeticOperation(Enum):
    """Enum for operations that can be performed on operators."""

    ADDITION = "sum"
    MULTIPLICATION = "prod"


def _compute_sum_shapes(
    operands: list[Operator],
) -> tuple[tuple[int, ...] | None, tuple[int, ...] | None]:
    input_shape = None
    output_shape = None

    for op in operands:
        if op.input_shape is not None:
            if input_shape is not None and input_shape != op.input_shape:
                raise ValueError(
                    f"Input shape mismatch in sum: expected {input_shape}, "
                    f"but {op} has input shape {op.input_shape}"
                )
            input_shape = op.input_shape

        if op.output_shape is not None:
            if output_shape is not None and output_shape != op.output_shape:
                raise ValueError(
                    f"Output shape mismatch in sum: expected {output_shape}, "
                    f"but {op} has output shape {op.output_shape}"
                )
            output_shape = op.output_shape

    return input_shape, output_shape


def _compute_product_shapes(
    operands: list[Operator],
) -> tuple[tuple[int, ...] | None, tuple[int, ...] | None]:
    for i in range(len(operands) - 1):
        left, right = operands[i], operands[i + 1]
        if left.input_shape is not None and right.output_shape is not None:
            if left.input_shape != right.output_shape:
                raise ValueError(
                    f"Shape mismatch in composition: {left} expects input "
                    f"{left.input_shape}, but {right} produces {right.output_shape}"
                )

    input_shape = operands[-1].input_shape
    output_shape = operands[0].output_shape
    return input_shape, output_shape


class Operator:
    """Base class for all operators."""

    def __init__(
        self,
        name: str | None = None,
        scalar: Numeric = 1,
        state: dict[str, Any] | None = None,
        arithmetic_operation: OperatorArithmeticOperation | None = None,
        operands: list[Operator] | None = None,
        input_shape: tuple[int, ...] | None = None,
        output_shape: tuple[int, ...] | None = None,
        adjoint: bool = False,
    ):
        if name is None:
            name = self.__class__.__name__
        self.name = name

        if state is None:
            state = {}
        self.state = state

        self._arithmetic_operation = arithmetic_operation
        if operands is None:
            operands = []
        self._operands = operands

        self._scalar: Numeric = 1
        self.scalar = scalar

        self.input_shape = input_shape
        self.output_shape = output_shape
        self._adjoint = adjoint

    def _scalar_repr_(self) -> str:
        scalar_repr = ""
        if self.scalar != 1:
            scalar_repr = repr(self.scalar)
            if self.scalar < 0:
                scalar_repr = f"({scalar_repr})"
        return scalar_repr

    def _composite_repr_(self) -> str:
        """Representation of a composite operator."""
        assert self.is_composite(), "This method is for composite operators only."
        scalar_repr = self._scalar_repr_()
        if self._arithmetic_operation == OperatorArithmeticOperation.ADDITION:
            op_repr = " + ".join(repr(op) for op in self._operands)
            if scalar_repr:
                return f"{scalar_repr}*({op_repr})"
            return op_repr
        elif self._arithmetic_operation == OperatorArithmeticOperation.MULTIPLICATION:
            op_reprs = []
            for op in self._operands:
                if op.is_composite():
                    op_reprs.append(f"({repr(op)})")
                else:
                    op_reprs.append(repr(op))
            op_repr = "*".join(op_reprs)
            if scalar_repr:
                return f"{scalar_repr}*{op_repr}"
            return op_repr
        raise ValueError(f"Unknown arithmetic operation: {self._arithmetic_operation}")

    def _repr_name_(self) -> str:
        """Return the operator name used in string representations."""
        return self.name

    def __repr__(self) -> str:
        if not self.is_composite():
            scalar_repr = self._scalar_repr_()
            name = self._repr_name_()
            if scalar_repr:
                return f"{scalar_repr}*{name}"
            return name
        return self._composite_repr_()

    def __eq__(self, other: Any) -> bool:
        """Check equality of two operators."""
        if not isinstance(other, Operator):
            return False

        return all(
            [
                type(self) is type(other),
                self.name == other.name,
                self.scalar == other.scalar,
                self.state == other.state,
                self._arithmetic_operation == other._arithmetic_operation,
                self._operands == other._operands,
                self.input_shape == other.input_shape,
                self.output_shape == other.output_shape,
                self.adjoint == other.adjoint,
            ]
        )

    @property
    def adjoint(self) -> bool:
        """Whether this operator represents its adjoint action."""
        return self._adjoint

    @property
    def T(self) -> Operator:
        """Return the adjoint operator.

        Composite adjoints follow the usual algebraic rules:
        ``(A + B).T = A.T + B.T`` and ``(A * B).T = B.T * A.T``.
        """
        if self.is_composite():
            if self._arithmetic_operation == OperatorArithmeticOperation.ADDITION:
                adjoint = self._operands[0].T
                for operand in self._operands[1:]:
                    adjoint = adjoint + operand.T
                return np.conjugate(self.scalar) * adjoint

            if self._arithmetic_operation == OperatorArithmeticOperation.MULTIPLICATION:
                operands = [operand.T for operand in reversed(self._operands)]
                adjoint = operands[0]
                for operand in operands[1:]:
                    adjoint = adjoint * operand
                return np.conjugate(self.scalar) * adjoint

            raise ValueError(
                f"Unknown arithmetic operation: {self._arithmetic_operation}"
            )

        operator_copy = deepcopy(self)
        operator_copy._adjoint = not self._adjoint
        operator_copy.scalar = np.conjugate(operator_copy.scalar)
        operator_copy.input_shape, operator_copy.output_shape = (
            operator_copy.output_shape,
            operator_copy.input_shape,
        )
        return operator_copy

    @property
    def scalar(self) -> Numeric:
        return self._scalar

    @scalar.setter
    def scalar(self, value: Numeric):
        """Set the scalar value of the operator."""
        if self.is_composite():
            if self._arithmetic_operation == OperatorArithmeticOperation.ADDITION:
                for child_operator in self._operands:
                    child_operator.scalar = child_operator.scalar * value
            elif self._arithmetic_operation == OperatorArithmeticOperation.MULTIPLICATION:
                self._operands[0].scalar = self._operands[0].scalar * value
        else:
            self._scalar = value

    def norm_estimate(
        self,
        number_iterations: int = 50,
        dtype: npt.DTypeLike = np.float32,
        input_shape: tuple[int, ...] | None = None,
        algorithm: Literal["poweriteration", "naive"] = "poweriteration",
        queue: Any | None = None,
        rng: np.random.Generator | None = None,
        **kwargs: Any,
    ) -> float:
        """Estimate the operator norm.

        Parameters
        ----------
        number_iterations:
            Positive number of power-iteration updates.
        dtype:
            Data type of the random starting vector.
        input_shape:
            Shape of the starting vector. Defaults to :attr:`input_shape`.
        algorithm:
            ``"poweriteration"`` applies power iteration to ``self.T * self``.
            ``"naive"`` combines power-iteration estimates for composite leaves
            using the triangle inequality for sums and submultiplicativity for
            products. Since the leaf norms are themselves estimates, the result
            is a heuristic and not a guaranteed upper bound.
        queue:
            Optional backend-specific execution queue forwarded to operator
            applications.
        rng:
            Random-number generator used to construct the starting vector.
        """
        if algorithm not in {"poweriteration", "naive"}:
            raise ValueError(
                "Unknown norm-estimation algorithm. Expected 'poweriteration' or 'naive'."
            )
        if (
            not isinstance(number_iterations, Integral)
            or isinstance(number_iterations, bool)
            or number_iterations < 1
        ):
            raise ValueError("number_iterations must be a positive integer")
        number_iterations = int(number_iterations)

        exact_norm = self._exact_norm()
        if exact_norm is not None:
            return exact_norm

        if algorithm == "naive" and self.is_composite():
            child_norms = [
                operand.norm_estimate(
                    number_iterations=number_iterations,
                    dtype=dtype,
                    algorithm=algorithm,
                    queue=queue,
                    rng=rng,
                    **kwargs,
                )
                for operand in self._operands
            ]
            scalar_magnitude = abs(float(self.scalar))
            if self._arithmetic_operation == OperatorArithmeticOperation.ADDITION:
                return scalar_magnitude * sum(child_norms)
            if self._arithmetic_operation == OperatorArithmeticOperation.MULTIPLICATION:
                return scalar_magnitude * prod(child_norms)
            raise ValueError(
                f"Unknown arithmetic operation: {self._arithmetic_operation}"
            )

        return self._power_iteration_norm_estimate(
            number_iterations=number_iterations,
            dtype=dtype,
            input_shape=input_shape,
            queue=queue,
            rng=rng,
            **kwargs,
        )

    def _exact_norm(self) -> float | None:
        """Return an exact norm when one is available for this operator."""
        return None

    def _power_iteration_norm_estimate(
        self,
        number_iterations: int,
        dtype: npt.DTypeLike,
        input_shape: tuple[int, ...] | None,
        queue: Any | None,
        rng: np.random.Generator | None,
        **kwargs: Any,
    ) -> float:
        if input_shape is None:
            input_shape = self.input_shape
        if input_shape is None:
            raise ValueError(
                "Cannot estimate the norm without an input shape. Set "
                "operator.input_shape or pass input_shape explicitly."
            )

        if rng is None:
            rng = np.random.default_rng()

        x = np.asarray(rng.standard_normal(input_shape), dtype=dtype)
        x_norm = self._vector_norm(x)
        if x_norm == 0:
            return 0.0
        x = x / x_norm

        estimate = 0.0
        adjoint = self.T
        for _ in range(number_iterations):
            y = (
                self.apply_to(x, queue=queue, **kwargs)
                if queue is not None
                else self.apply_to(x, **kwargs)
            )
            z = (
                adjoint.apply_to(y, queue=queue, **kwargs)
                if queue is not None
                else adjoint.apply_to(y, **kwargs)
            )
            z_norm = adjoint._vector_norm(z)
            if z_norm == 0:
                return 0.0

            # Since x is normalized, ||(A.T * A)x|| converges to the dominant
            # eigenvalue of A.T * A. Its square root is the operator norm. This
            # remains valid when T represents an adjoint for weighted inner
            # products, as is the case for gratopy's projection operators.
            estimate = float(np.sqrt(z_norm))
            x = z / z_norm

        return estimate

    def _vector_norm(self, vector: Any) -> float:
        """Compute a vector norm using this operator's array backend."""
        if self.is_composite():
            return self._operands[0]._vector_norm(vector)
        return float(np.linalg.norm(vector))

    def apply_to(
        self,
        argument: Any,
        output: Any | None = None,
        **kwargs: Any,
    ) -> Any:
        """Application of this operator to some given argument.

        For composite products, ``output`` is passed only to the final child
        operator. Intermediate results are computed normally while the final
        result can reuse the caller-provided array.
        """
        if self.is_composite():
            if self._arithmetic_operation == OperatorArithmeticOperation.ADDITION:
                if output is None:
                    result = self._operands[0].apply_to(argument, **kwargs)
                    for child_op in self._operands[1:]:
                        result += child_op.apply_to(argument, **kwargs)
                    return result

                result = self._operands[0].apply_to(argument, output=output, **kwargs)
                for child_op in self._operands[1:]:
                    result += child_op.apply_to(argument, **kwargs)
                return result
            elif self._arithmetic_operation == OperatorArithmeticOperation.MULTIPLICATION:
                result = argument
                application_order = list(reversed(self._operands))
                for index, child_op in enumerate(application_order):
                    child_output = output if index == len(application_order) - 1 else None
                    result = child_op.apply_to(result, output=child_output, **kwargs)
                return result

        raise NotImplementedError(
            "apply_to needs to be implemented in specialized subclasses"
        )

    def is_composite(self) -> bool:
        """Check if the operator is composite."""
        return self._arithmetic_operation is not None

    def __add__(self, other: Operator) -> Operator:
        """Add another operator to this one."""
        if not isinstance(other, Operator):
            raise TypeError(f"Cannot add {type(other)} to {type(self)}")

        if isinstance(other, _ZeroOperator):
            return self

        operands = []
        for operator in [deepcopy(self), deepcopy(other)]:
            if (
                operator.is_composite()
                and operator._arithmetic_operation == OperatorArithmeticOperation.ADDITION
            ):
                for child_operator in operator._operands:
                    child_operator.scalar *= operator.scalar
                    operands.append(child_operator)
            else:
                operands.append(operator)

        input_shape, output_shape = _compute_sum_shapes(operands)

        return Operator(
            name=None,
            scalar=1,
            arithmetic_operation=OperatorArithmeticOperation.ADDITION,
            operands=operands,
            input_shape=input_shape,
            output_shape=output_shape,
        )

    def __neg__(self) -> Operator:
        """Negate this operator."""
        return self.__rmul__(-1)

    def __sub__(self, other: Operator) -> Operator:
        """Subtract another operator from this one."""
        return self + (-other)

    def __rmul__(self, other: Numeric) -> Operator:  # type: ignore[misc]
        """Right-multiply this operator by a scalar."""
        if other == 0:
            return ZERO
        if other == 1:
            return self

        operator_copy = deepcopy(self)
        operator_copy.scalar = operator_copy.scalar * other
        return operator_copy

    def __mul__(self, other: Operator | Any) -> Operator | Any:
        """Multiply this operator by another operator, or apply it to appropriate input."""
        if not isinstance(other, Operator):
            if isinstance(other, Number):
                return self.__rmul__(other)  # type: ignore[arg-type, operator]

            # attempt to apply the operator to the input
            return self.apply_to(other)

        if isinstance(other, _ZeroOperator):
            return other

        if isinstance(other, _IdentityOperator):
            return self

        operands = []
        scalar: Numeric = 1
        for operator in [deepcopy(self), deepcopy(other)]:
            if (
                operator.is_composite()
                and operator._arithmetic_operation
                == OperatorArithmeticOperation.MULTIPLICATION
            ):
                operands.extend(operator._operands)
            else:
                scalar *= operator.scalar
                operator.scalar = 1
                operands.append(operator)

        input_shape, output_shape = _compute_product_shapes(operands)

        return Operator(
            name=None,
            scalar=scalar,
            arithmetic_operation=OperatorArithmeticOperation.MULTIPLICATION,
            operands=operands,
            input_shape=input_shape,
            output_shape=output_shape,
        )


class _IdentityOperator(Operator):
    """Base class for identity operator."""

    @property
    def T(self) -> Operator:
        """The identity operator is self-adjoint."""
        return self

    def __mul__(self, other: Operator | Any) -> Operator | Any:
        """Multiplying the identity operator with another operator returns
        the other operator."""
        if isinstance(other, Operator):
            return other
        return super().__mul__(other)

    def _exact_norm(self) -> float:
        """Return the exact norm of the identity operator."""
        return abs(float(self.scalar))

    def apply_to(self, argument: Any, output: Any | None = None, **kwargs: Any) -> Any:
        """The identity operator does not change the input."""
        result = self.scalar * argument
        if output is not None:
            output[...] = result
            return output
        return result


class _ZeroOperator(Operator):
    """Base class for zero operator."""

    @property
    def T(self) -> Operator:
        """The zero operator is self-adjoint."""
        # This singleton zero operator is shape-agnostic. If we later introduce
        # shape-aware zero operators, their adjoints should swap input and
        # output shapes just like regular operators.
        return self

    def __add__(self, other: Operator) -> Operator:
        """Adding zero operator to any operator returns the other operator."""
        return other

    @property
    def scalar(self) -> Numeric:
        return self._scalar

    @scalar.setter
    def scalar(self, value: Numeric):
        pass

    def _exact_norm(self) -> float:
        """Return the exact norm of the zero operator."""
        return 0.0

    def apply_to(self, argument: Any, output: Any | None = None, **kwargs: Any) -> Any:
        """Applying the zero operator returns a zero-multiplied version of the input."""
        if output is not None:
            output[...] = 0
            return output

        try:
            return 0 * argument  # type: ignore
        except TypeError:
            pass

        try:
            return np.zeros_like(argument)
        except (ValueError, TypeError):
            pass

        raise TypeError(f"Cannot apply zero operator to {type(argument)}")


IDENTITY = _IdentityOperator(name="[Id]")
ZERO = _ZeroOperator(name="[0]")
