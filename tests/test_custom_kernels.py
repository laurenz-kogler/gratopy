from __future__ import annotations

import gc
import weakref
from threading import Thread

import numpy as np
import pyopencl as cl
import pyopencl.array as clarray

from pathlib import Path

from gratopy.operator import invalidate_kernel_cache
from gratopy.operator.opencl import OpenCLKernelSpec, _OpenCLOperator, _PROGRAM_CACHE


TEST_AFFINE_KERNEL = Path(__file__).parent / "affine.cl"
TEST_AFFINE_ALT_KERNEL = Path(__file__).parent / "affine_alt.cl"


class AffineOperator(_OpenCLOperator):
    def __init__(
        self,
        shape: tuple[int, ...],
        kernel_spec: OpenCLKernelSpec | None = None,
    ):
        super().__init__(
            name="AffineOperator",
            input_shape=shape,
            output_shape=shape,
            kernel_spec=kernel_spec,
        )

    def _default_kernel_spec(self) -> OpenCLKernelSpec:
        return OpenCLKernelSpec.from_path(TEST_AFFINE_KERNEL, base_name="affine")


def test_custom_kernel_spec_end_to_end():
    ctx = cl.create_some_context(interactive=False)
    queue = cl.CommandQueue(ctx)

    shape = (4, 5)
    spec = OpenCLKernelSpec.from_path(TEST_AFFINE_KERNEL, base_name="affine")
    operator = AffineOperator(shape=shape, kernel_spec=spec)

    expected = 2 * np.arange(np.prod(shape), dtype=np.float32).reshape(shape) + 1

    for input_order in ["C", "F"]:
        for output_order in ["C", "F"]:
            host_input = np.require(
                np.arange(np.prod(shape), dtype=np.float32).reshape(shape),
                requirements=input_order,
            )
            device_input = clarray.to_device(queue, host_input)
            device_output = clarray.zeros(
                queue, shape, dtype=np.float32, order=output_order
            )

            result = operator.apply_to(device_input, output=device_output)
            np.testing.assert_allclose(result.get(), expected)


def test_custom_kernel_spec_adjoint_kernel_is_used():
    ctx = cl.create_some_context(interactive=False)
    queue = cl.CommandQueue(ctx)

    shape = (4, 5)
    spec = OpenCLKernelSpec.from_path(TEST_AFFINE_KERNEL, base_name="affine")
    operator = AffineOperator(shape=shape, kernel_spec=spec)

    host_input = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    expected = 3 * host_input - 1

    result = operator.T.apply_to(host_input, queue=queue)
    np.testing.assert_allclose(result.get(), expected)


def test_program_bundle_is_shared_while_operators_are_alive():
    invalidate_kernel_cache()
    ctx = cl.create_some_context(interactive=False)
    queue = cl.CommandQueue(ctx)
    shape = (4, 5)
    op1 = AffineOperator(shape=shape)
    op2 = AffineOperator(shape=shape)

    op1.apply_to(np.ones(shape, dtype=np.float32), queue=queue)
    op2.apply_to(np.ones(shape, dtype=np.float32), queue=queue)

    lease_key = (ctx, True)
    bundle1 = op1._program_bundles[lease_key]
    bundle2 = op2._program_bundles[lease_key]
    bundle_ref = weakref.ref(bundle1)

    assert bundle1 is bundle2
    assert len(_PROGRAM_CACHE) == 1

    del bundle1, bundle2, op1
    gc.collect()
    assert bundle_ref() is not None

    del op2
    gc.collect()
    assert bundle_ref() is None
    assert len(_PROGRAM_CACHE) == 0


def test_program_cache_invalidation_replaces_live_operator_lease():
    invalidate_kernel_cache()
    ctx = cl.create_some_context(interactive=False)
    operator = AffineOperator(shape=(4, 5))

    old_bundle = operator._get_program(ctx)
    invalidate_kernel_cache()

    assert len(_PROGRAM_CACHE) == 0

    new_bundle = operator._get_program(ctx)

    assert new_bundle is not old_bundle
    assert new_bundle.generation > old_bundle.generation
    assert len(_PROGRAM_CACHE) == 1


def test_program_bundle_uses_thread_local_kernel_instances():
    invalidate_kernel_cache()
    ctx = cl.create_some_context(interactive=False)
    operator = AffineOperator(shape=(4, 5))
    bundle = operator._get_program(ctx)
    kernel_name = operator._kernel_name(
        dtype=np.float32,
        output_order="F",
        input_order="F",
    )
    main_kernel = bundle.kernel(kernel_name)
    worker_kernels = []

    thread = Thread(target=lambda: worker_kernels.append(bundle.kernel(kernel_name)))
    thread.start()
    thread.join()

    assert bundle.kernel(kernel_name) is main_kernel
    assert worker_kernels[0] is not main_kernel


def test_custom_kernel_spec_cache_isolation_by_source_signature():
    ctx = cl.create_some_context(interactive=False)
    queue = cl.CommandQueue(ctx)

    shape = (4, 5)
    host_input = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)

    spec1 = OpenCLKernelSpec.from_path(TEST_AFFINE_KERNEL, base_name="affine")
    spec2 = OpenCLKernelSpec.from_path(TEST_AFFINE_ALT_KERNEL, base_name="affine")

    op1 = AffineOperator(shape=shape, kernel_spec=spec1)
    op2 = AffineOperator(shape=shape, kernel_spec=spec2)

    expected1 = 2 * host_input + 1
    expected2 = 5 * host_input - 3

    result1_before = op1.apply_to(host_input, queue=queue).get()
    result2 = op2.apply_to(host_input, queue=queue).get()
    result1_after = op1.apply_to(host_input, queue=queue).get()

    np.testing.assert_allclose(result1_before, expected1)
    np.testing.assert_allclose(result2, expected2)
    np.testing.assert_allclose(result1_after, expected1)
