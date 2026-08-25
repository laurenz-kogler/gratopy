"""Gratopy projection operators.

This module contains concrete operator implementations for gratopy's
experimental operator interface, most notably :class:`.Radon`.

The focus is currently on a compositional, operator-style interface for Radon
transforms and their adjoints. Fanbeam support is not yet implemented at the
same level.

Examples
--------

>>> import numpy as np
>>> import pyopencl as cl
>>> import gratopy
>>> ctx = cl.create_some_context(interactive=False)
>>> queue = cl.CommandQueue(ctx)
>>> Nx = 300
>>> img = np.zeros((Nx, Nx), dtype=np.float32)
>>> R = gratopy.operator.Radon(image_domain=Nx, angles=180)
>>> sinogram = R.apply_to(img, queue=queue)
>>> backprojection = R.T.apply_to(sinogram)

The same forward and adjoint applications can also be written via operator
syntax:

>>> sinogram = R * img
>>> backprojection = R.T * sinogram
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pyopencl as cl
import pyopencl.array as clarray

from dataclasses import replace
from pathlib import Path
from typing import Any

from gratopy.gratopy import radon_struct
from gratopy.operator.base import Operator
from gratopy.operator.opencl import OpenCLKernelSpec, _OpenCLOperator
from gratopy.utilities import (
    Angles,
    Detectors,
    ExtentPlaceholder,
    ImageDomain,
    full_detector_given_image_fullcircle,
    full_detector_given_image_halfcircle,
    full_image_given_detector_fullcircle,
    full_image_given_detector_halfcircle,
    valid_detector_given_image_fullcircle,
    valid_detector_given_image_halfcircle,
    valid_image_given_detector_fullcircle,
    valid_image_given_detector_halfcircle,
)


class Radon(_OpenCLOperator):
    """Parallel-beam Radon transform operator.

    This class provides the main entry point to gratopy's experimental
    operator-based projection interface. A :class:`Radon` object represents
    the forward projection operator. Accessing :attr:`T` creates an adjoint
    expression wrapper that references the same :class:`Radon` instance and
    shares its runtime caches.

    **Parameters**

    ``image_domain``:
        Description of the image grid and its physical extent. This can be
        given as:

        - an ``int`` for a square image domain of that size,
        - a ``(Nx, Ny)`` tuple for a rectangular grid,
        - or an explicit :class:`gratopy.utilities.ImageDomain` instance.

        Plain integer and tuple inputs are converted to
        :class:`~gratopy.utilities.ImageDomain` with a default extent of
        ``2.0``.
    ``angles``:
        Angular sampling of the operator. This can be given either as an
        integer or as an explicit :class:`gratopy.utilities.Angles` object.

        If an integer is given, uniformly weighted angles are created via
        :meth:`gratopy.utilities.Angles.uniform` with the ``half_circle``
        parameter set to ``True``.
    ``detectors``:
        Detector configuration. This can be given as:

        - ``None`` to use the default detector count
          ``ceil(hypot(Nx, Ny))``,
        - an integer specifying the number of detector pixels,
        - or an explicit :class:`gratopy.utilities.Detectors` object.

        Plain integer inputs are converted to
        :class:`~gratopy.utilities.Detectors` with default extent handling.
    ``kernel_spec``:
        Optional :class:`gratopy.operator.opencl.OpenCLKernelSpec` describing
        which OpenCL kernel bundle to use. When omitted, the operator uses the
        Radon kernels shipped with gratopy.

    **Notes**

    The operator accepts both :class:`pyopencl.array.Array` inputs and NumPy
    arrays. When applying the operator to a NumPy array, a queue must be
    available so that the data can be transferred to the device.

    The operator supports 2D as well as slicewise 3D data. For example,
    a forward operator with image shape ``(Nx, Ny)`` maps:

    - ``(Nx, Ny)`` to ``(Ns, Na)``,
    - ``(Nx, Ny, Nz)`` to ``(Ns, Na, Nz)``.

    Correspondingly, the adjoint maps:

    - ``(Ns, Na)`` to ``(Nx, Ny)``,
    - ``(Ns, Na, Nz)`` to ``(Nx, Ny, Nz)``.

    Extent placeholders are supported experimentally for Radon geometry when
    exactly one extent is fixed numerically and the other is given as
    :class:`gratopy.utilities.ExtentPlaceholder`. The operator can resolve an
    image extent from a fixed detector extent, or a detector extent from a
    fixed image extent. Passing placeholders for both extents at once remains
    unsupported and raises :class:`NotImplementedError`; geometries for which
    no valid placeholder resolution exists raise :class:`ValueError`.

    **Examples**

    >>> import numpy as np
    >>> import pyopencl as cl
    >>> import gratopy
    >>> ctx = cl.create_some_context(interactive=False)
    >>> queue = cl.CommandQueue(ctx)
    >>> img = np.zeros((128, 128), dtype=np.float32)
    >>> R = gratopy.operator.Radon(image_domain=128, angles=180)
    >>> sino = R.apply_to(img, queue=queue)
    >>> backproj = R.T.apply_to(sino)

    Operator algebra is also supported:

    >>> G = R.T * R
    >>> gram_img = G.apply_to(img, queue=queue)
    """

    def __init__(
        self,
        image_domain: int | tuple[int, int] | ImageDomain,
        angles: Angles | int,
        detectors: Detectors | int | None = None,
        kernel_spec: OpenCLKernelSpec | None = None,
    ):
        if not isinstance(image_domain, ImageDomain):
            image_domain = ImageDomain(size=image_domain, extent=2.0)

        if not isinstance(angles, Angles):
            angles = Angles.uniform(number=angles, half_circle=True)

        if not isinstance(detectors, Detectors):
            if detectors is None:
                detectors = int(np.ceil(np.hypot(*image_domain.size)))
            detector_extent = image_domain.extent
            if isinstance(detector_extent, ExtentPlaceholder):
                detector_extent = ExtentPlaceholder.FULL
            detectors = Detectors(number=detectors, extent=detector_extent)

        state = {
            "image_domain": image_domain,
            "angles": angles,
            "detectors": detectors,
        }
        super().__init__(
            name="Radon",
            state=state,
            kernel_spec=kernel_spec,
        )

        self._resolve_extent_placeholders()
        self._host_struct: dict[str, Any] | None = None
        self._device_struct: dict[tuple[cl.Context, np.dtype], dict[str, Any]] = {}

        image_shape = self.image_domain.size
        sinogram_shape = (self.detectors.number, len(self.angles))

        self.input_shape = image_shape
        self.output_shape = sinogram_shape

    def _default_kernel_spec(self) -> OpenCLKernelSpec:
        kernel_path = Path(__file__).resolve().parent.parent / "radon.cl"
        return OpenCLKernelSpec.from_path(kernel_path, base_name="radon")

    def __getstate__(self) -> dict[str, Any]:
        """Return pickle/deepcopy state without live OpenCL runtime objects."""
        state = super().__getstate__()
        state["_host_struct"] = None
        state["_device_struct"] = {}
        return state

    @property
    def image_domain(self) -> ImageDomain:
        return self.state["image_domain"]

    @property
    def angles(self) -> Angles:
        return self.state["angles"]

    @property
    def detectors(self) -> Detectors:
        return self.state["detectors"]

    def _use_full_circle(self) -> bool:
        """Decide whether extent placeholders use full-circle geometry.

        The choice follows the angular sampling's ``half_circle`` flag. That
        flag is load-bearing only for the equispaced constructors
        (:meth:`Angles.sparse` / :meth:`Angles.uniform`); for limited-angle and
        explicit samplings it is plain metadata defaulting to ``False``. So a
        placeholder resolved on a sampling that was never explicitly flagged
        could silently fall back to the full-circle formulas. Guard against
        that: a full-circle resolution is only trusted when the sampled angles
        actually cover more than half a circle.
        """
        if self.angles.half_circle:
            return False

        angles = np.asarray(self.angles.angles, dtype=float)
        angular_range = float(angles.max() - angles.min())
        # POLICY (review): "covers the full circle" == angular range > pi.
        # This cleanly separates half-circle data (range < pi) from full-circle
        # data (range ~= 2*pi); a partial scan in between is treated as full.
        if angular_range <= np.pi + 1e-9:
            raise ValueError(
                "Cannot resolve an ExtentPlaceholder: the angular sampling is "
                "marked half_circle=False (full circle), but the angles only "
                f"span {angular_range:.4f} rad (<= pi). If this is half-circle "
                "data, build the Angles with half_circle=True; if it is a "
                "genuine full-circle scan, widen the angular range."
            )
        return True

    def _resolve_extent_placeholders(self) -> None:
        """Resolve extent placeholders without mutating the input geometry."""
        image_extent = self.image_domain.extent
        detector_extent = self.detectors.extent
        if isinstance(image_extent, ExtentPlaceholder) and isinstance(
            detector_extent, ExtentPlaceholder
        ):
            raise NotImplementedError(
                "Both the ImageDomain and Detectors use an ExtentPlaceholder. "
                "Please set at least one of the two extents to a specific "
                "value; the remaining placeholder will be resolved "
                "automatically."
            )

        if isinstance(image_extent, ExtentPlaceholder):
            assert not isinstance(detector_extent, ExtentPlaceholder)
            Mx, My = self.image_domain.center
            Md = self.detectors.center
            Nx, Ny = self.image_domain.size
            c = Nx / Ny
            Dd = float(detector_extent)
            M = (Md, Mx, My)
            full_circle = self._use_full_circle()

            if image_extent == ExtentPlaceholder.FULL:
                if full_circle:
                    image_result = full_image_given_detector_fullcircle(M, Dd, c)
                else:
                    image_result = full_image_given_detector_halfcircle(M, Dd, c)
            else:
                if full_circle:
                    image_result = valid_image_given_detector_fullcircle(M, Dd, c)
                else:
                    image_result = valid_image_given_detector_halfcircle(M, Dd, c)

            if image_result is None:
                if image_extent == ExtentPlaceholder.FULL:
                    meaning = (
                        "FULL means the largest image extent such that every "
                        "ray through the image also hits the detector"
                    )
                else:
                    meaning = (
                        "VALID means the smallest image extent such that "
                        "every ray hitting the detector also passes through "
                        "the image"
                    )
                raise ValueError(
                    f"Cannot resolve ExtentPlaceholder.{image_extent.name} "
                    f"for the image domain ({meaning}): no such image extent "
                    f"exists with the given detector width Dd={Dd}, detector "
                    f"center Md={Md}, and image center ({Mx}, {My}). "
                    f"Consider increasing the detector width or adjusting the "
                    f"center offsets."
                )
            Dx, Dy = image_result
            self.state["image_domain"] = replace(
                self.image_domain,
                extent=max(Dx, Dy),
            )

        if isinstance(detector_extent, ExtentPlaceholder):
            assert not isinstance(image_extent, ExtentPlaceholder)
            Mx, My = self.image_domain.center
            Md = self.detectors.center
            Nx, Ny = self.image_domain.size
            extent = float(image_extent)
            Dx = extent * Nx / max(Nx, Ny)
            Dy = extent * Ny / max(Nx, Ny)
            M = (Md, Mx, My)
            D = (Dx, Dy)
            full_circle = self._use_full_circle()
            detector_result: float | None

            if detector_extent == ExtentPlaceholder.FULL:
                if full_circle:
                    detector_result = full_detector_given_image_fullcircle(M, D)
                else:
                    detector_result = full_detector_given_image_halfcircle(M, D)
            else:
                if full_circle:
                    detector_result = valid_detector_given_image_fullcircle(M, D)
                else:
                    detector_result = valid_detector_given_image_halfcircle(M, D)

            if detector_result is None:
                if detector_extent == ExtentPlaceholder.FULL:
                    meaning = (
                        "FULL means the smallest detector width such that "
                        "every ray through the image also hits the detector"
                    )
                else:
                    meaning = (
                        "VALID means the largest detector width such that "
                        "every ray hitting the detector also passes through "
                        "the image"
                    )
                raise ValueError(
                    f"Cannot resolve ExtentPlaceholder.{detector_extent.name} "
                    f"for the detector ({meaning}): no such detector width "
                    f"exists with image dimensions ({Dx}, {Dy}), detector "
                    f"center Md={Md}, and image center ({Mx}, {My}). "
                    f"Consider increasing the image extent or adjusting the "
                    f"center offsets."
                )
            self.state["detectors"] = replace(
                self.detectors,
                extent=detector_result,
            )

    def _ensure_host_struct(self, queue: cl.CommandQueue) -> None:
        if self._host_struct is not None:
            return

        self._host_struct = radon_struct(
            queue=queue,
            img_shape=self.image_domain.size,
            angles=self.angles.angles,
            angle_weights=self.angles.weights,
            n_detectors=self.detectors.number,
            detector_width=float(self.detectors.extent),  # type: ignore[arg-type]
            image_width=float(self.image_domain.extent),  # type: ignore[arg-type]
            midpoint_shift=self.image_domain.center,
            detector_shift=self.detectors.center,
        )

    def _ensure_device_struct(
        self,
        queue: cl.CommandQueue,
        dtype: npt.DTypeLike,
    ) -> dict[str, Any]:
        self._ensure_host_struct(queue)
        dtype = np.dtype(dtype)
        cache_key = (queue.context, dtype)
        if cache_key in self._device_struct:
            return self._device_struct[cache_key]

        assert self._host_struct is not None
        ofs = self._host_struct["ofs_dict"][dtype]
        geometry = self._host_struct["geo_dict"][dtype]

        ofs_buf = cl.Buffer(queue.context, cl.mem_flags.READ_ONLY, ofs.nbytes)
        geometry_buf = cl.Buffer(queue.context, cl.mem_flags.READ_ONLY, geometry.nbytes)

        cl.enqueue_copy(queue, ofs_buf, ofs.data).wait()
        cl.enqueue_copy(queue, geometry_buf, geometry.data).wait()

        device_struct = {
            "ofs": ofs_buf,
            "geometry": geometry_buf,
        }
        self._device_struct[cache_key] = device_struct
        return device_struct

    def _kernel_arguments(
        self,
        output: clarray.Array,
        argument: clarray.Array,
        queue: cl.CommandQueue,
    ) -> tuple[Any, ...]:
        device_struct = self._ensure_device_struct(queue, argument.dtype)
        return device_struct["ofs"], device_struct["geometry"]

    def apply_to(
        self,
        argument: Any,
        output: Any | None = None,
        queue: cl.CommandQueue | None = None,
        return_event: bool = False,
        **kwargs: Any,
    ) -> clarray.Array | tuple[clarray.Array, list[cl.Event]]:
        queue = self._infer_queue(argument=argument, output=output, queue=queue)
        return super().apply_to(
            argument,
            output=output,
            queue=queue,
            return_event=return_event,
        )

    def apply_adjoint_to(
        self,
        argument: Any,
        output: Any | None = None,
        queue: cl.CommandQueue | None = None,
        return_event: bool = False,
        **kwargs: Any,
    ) -> clarray.Array | tuple[clarray.Array, list[cl.Event]]:
        queue = self._infer_queue(argument=argument, output=output, queue=queue)
        return super().apply_adjoint_to(
            argument,
            output=output,
            queue=queue,
            return_event=return_event,
        )


class Fanbeam(Operator):
    def __init__(
        self,
        source_distances: float | tuple[float, float],
        image_domain: int | tuple[int, int] | ImageDomain,
        angles: Angles | int,
        detectors: Detectors | int | None = None,
    ):
        super().__init__(name="Fanbeam")


# R, R_E parameters for fanbeam tranform: pass as _one_ argument, tuple
# or new dataclass. source_detector_distance, source_origin_distance
# if only one value is given, it is R, and R_E is R/2.
