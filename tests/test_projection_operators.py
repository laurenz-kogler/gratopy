import numpy as np
import pytest
import pyopencl as cl
import pyopencl.array as clarray

from gratopy.operator import (
    Fanbeam,
    Radon,
    RayDrivenFanbeam,
    RayDrivenRadon,
    StripDrivenRadon,
)
from gratopy.operator.base import AdjointOperator
from gratopy.utilities import Angles, Detectors, ExtentPlaceholder, ImageDomain


PARALLEL_OPERATORS = (Radon, RayDrivenRadon, StripDrivenRadon)
FANBEAM_OPERATORS = (Fanbeam, RayDrivenFanbeam)


@pytest.fixture(scope="module")
def queue():
    context = cl.create_some_context(interactive=False)
    return cl.CommandQueue(context)


def make_parallel_operator(operator_class):
    return operator_class(
        image_domain=ImageDomain(size=(24, 20), extent=2.0, center=(0.05, -0.03)),
        angles=Angles.uniform(17, half_circle=True),
        detectors=Detectors(number=29, extent=3.0, center=0.1, reversed=True),
    )


def make_fanbeam_operator(operator_class):
    return operator_class(
        source_distances=(8.0, 4.0),
        image_domain=ImageDomain(size=(24, 20), extent=2.0, center=(0.05, -0.03)),
        angles=Angles.uniform(24),
        detectors=Detectors(number=29, extent=3.0, center=0.1, reversed=True),
    )


@pytest.mark.parametrize(
    ("operator_class", "expected_name", "kernel_base_name"),
    [
        (Radon, "Radon", "radon"),
        (RayDrivenRadon, "RayDrivenRadon", "radon_ray"),
        (StripDrivenRadon, "StripDrivenRadon", "radon_strip"),
    ],
)
def test_parallel_operator_variants_use_stable_leaf_architecture(
    operator_class,
    expected_name,
    kernel_base_name,
):
    operator = make_parallel_operator(operator_class)

    assert repr(operator) == expected_name
    assert operator.kernel_spec.base_name == kernel_base_name
    assert "adjoint" not in operator.state
    assert isinstance(operator.T, AdjointOperator)
    assert operator.T.operand is operator
    assert operator.T.T is operator


@pytest.mark.parametrize(
    ("operator_class", "expected_name", "kernel_base_name"),
    [
        (Fanbeam, "Fanbeam", "fanbeam"),
        (RayDrivenFanbeam, "RayDrivenFanbeam", "fanbeam_ray"),
    ],
)
def test_fanbeam_operator_variants_use_stable_leaf_architecture(
    operator_class,
    expected_name,
    kernel_base_name,
):
    operator = make_fanbeam_operator(operator_class)

    assert repr(operator) == expected_name
    assert operator.kernel_spec.base_name == kernel_base_name
    assert "adjoint" not in operator.state
    assert isinstance(operator.T, AdjointOperator)
    assert operator.T.operand is operator
    assert operator.T.T is operator


@pytest.mark.parametrize("operator_class", PARALLEL_OPERATORS)
@pytest.mark.parametrize("order", ["C", "F"])
def test_parallel_operator_variants_execute_slicewise(operator_class, order, queue):
    operator = make_parallel_operator(operator_class)
    rng = np.random.default_rng(1)
    image = np.require(
        rng.standard_normal(operator.input_shape + (2,)),
        dtype=np.float32,
        requirements=order,
    )
    device_image = clarray.to_device(queue, image)

    sinogram = operator.apply_to(device_image)
    backprojection = operator.T.apply_to(sinogram)

    assert sinogram.shape == operator.output_shape + (2,)
    assert backprojection.shape == operator.input_shape + (2,)
    assert np.all(np.isfinite(sinogram.get()))
    assert np.all(np.isfinite(backprojection.get()))


@pytest.mark.parametrize("operator_class", FANBEAM_OPERATORS)
@pytest.mark.parametrize("order", ["C", "F"])
def test_fanbeam_operator_variants_execute_slicewise(operator_class, order, queue):
    operator = make_fanbeam_operator(operator_class)
    rng = np.random.default_rng(1)
    image = np.require(
        rng.standard_normal(operator.input_shape + (2,)),
        dtype=np.float32,
        requirements=order,
    )
    device_image = clarray.to_device(queue, image)

    sinogram = operator.apply_to(device_image)
    backprojection = operator.T.apply_to(sinogram)

    assert sinogram.shape == operator.output_shape + (2,)
    assert backprojection.shape == operator.input_shape + (2,)
    assert np.all(np.isfinite(sinogram.get()))
    assert np.all(np.isfinite(backprojection.get()))


def assert_physical_adjointness(operator, queue):
    rng = np.random.default_rng(2)
    image = clarray.to_device(
        queue,
        rng.standard_normal(operator.input_shape).astype(np.float32),
    )
    sinogram = clarray.to_device(
        queue,
        rng.standard_normal(operator.output_shape).astype(np.float32),
    )

    projected = operator.apply_to(image).get()
    backprojected = operator.T.apply_to(sinogram).get()
    image_values = image.get()
    sinogram_values = sinogram.get()

    image_extent = operator.image_domain.extent
    detector_extent = operator.detectors.extent
    assert not isinstance(image_extent, ExtentPlaceholder)
    assert not isinstance(detector_extent, ExtentPlaceholder)
    delta_x = image_extent / max(operator.image_domain.size)
    delta_s = detector_extent / operator.detectors.number

    image_pairing = np.vdot(image_values, backprojected) * delta_x**2
    weighted_sinogram = sinogram_values * operator.angles.weights[np.newaxis, :]
    sinogram_pairing = np.vdot(weighted_sinogram, projected) * delta_s

    assert image_pairing == pytest.approx(sinogram_pairing, rel=2e-4, abs=2e-5)


@pytest.mark.parametrize("operator_class", PARALLEL_OPERATORS)
def test_parallel_operator_variants_are_physical_adjoints(operator_class, queue):
    assert_physical_adjointness(make_parallel_operator(operator_class), queue)


@pytest.mark.parametrize("operator_class", FANBEAM_OPERATORS)
def test_fanbeam_operator_variants_are_physical_adjoints(operator_class, queue):
    assert_physical_adjointness(make_fanbeam_operator(operator_class), queue)


@pytest.mark.parametrize("operator_class", PARALLEL_OPERATORS)
def test_parallel_operator_variants_share_runtime_with_adjoint(operator_class, queue):
    operator = make_parallel_operator(operator_class)
    image = clarray.zeros(queue, operator.input_shape, dtype=np.float32)

    for _ in range(3):
        (operator.T * operator).apply_to(image)
    queue.finish()

    assert len(operator._device_struct) == 1
    assert set(next(iter(operator._device_struct.values()))) == {"ofs", "geometry"}


@pytest.mark.parametrize("operator_class", FANBEAM_OPERATORS)
def test_fanbeam_operator_variants_share_runtime_with_adjoint(operator_class, queue):
    operator = make_fanbeam_operator(operator_class)
    image = clarray.zeros(queue, operator.input_shape, dtype=np.float32)

    for _ in range(3):
        (operator.T * operator).apply_to(image)
    queue.finish()

    assert len(operator._device_struct) == 1
    assert set(next(iter(operator._device_struct.values()))) == {
        "ofs",
        "sdpd",
        "geometry",
    }


def test_fanbeam_scalar_source_distance_uses_half_distance_to_origin():
    operator = Fanbeam(source_distances=8.0, image_domain=16, angles=20)

    assert operator.source_detector_distance == 8.0
    assert operator.source_origin_distance == 4.0
    assert operator.angles.half_circle is False


@pytest.mark.parametrize("source_distances", [(4.0, 4.0), (3.0, 4.0), -2.0])
def test_fanbeam_rejects_invalid_source_distances(source_distances):
    with pytest.raises(ValueError, match="source_.*distance"):
        Fanbeam(source_distances=source_distances, image_domain=16, angles=20)


def test_fanbeam_rejects_extent_placeholders_without_mutating_geometry():
    image_domain = ImageDomain(size=16, extent=2.0)
    detectors = Detectors(number=20, extent=ExtentPlaceholder.FULL)

    with pytest.raises(NotImplementedError, match="numeric image and detector extents"):
        Fanbeam(
            source_distances=8.0,
            image_domain=image_domain,
            angles=20,
            detectors=detectors,
        )

    assert image_domain.extent == 2.0
    assert detectors.extent is ExtentPlaceholder.FULL
