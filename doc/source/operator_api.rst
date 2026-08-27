Operator syntax
===============

The operator API provides a more compositional interface to gratopy's
projection operators. It is currently **experimental** and supports both
parallel-beam and fan-beam transforms through several discretizations.

The legacy :class:`gratopy.ProjectionSettings` API remains the main documented
interface for the full feature set of gratopy. The operator API complements it
with a syntax that is often more convenient when working with operator algebra,
adjoint operators, and experimental kernels.

.. warning::

   The operator API is **experimental**. Backward-incompatible changes may be
   introduced without a full deprecation cycle while the interface and internal
   abstractions are still settling.

   Extent placeholders such as
   :class:`gratopy.utilities.ExtentPlaceholder` are supported experimentally
   for parallel-beam operators when exactly one of the image or detector
   extents is a placeholder. Fan-beam operators currently require numeric
   extents.

Current scope
-------------

The current operator API supports in particular:

- pixel-driven :class:`gratopy.operator.projection.Radon` and
  :class:`gratopy.operator.projection.Fanbeam` operators,
- :class:`gratopy.operator.projection.RayDrivenRadon`,
  :class:`gratopy.operator.projection.StripDrivenRadon`, and
  :class:`gratopy.operator.projection.RayDrivenFanbeam` variants,
- adjoints via :attr:`T`,
- operator composition and arithmetic,
- custom OpenCL kernels via :class:`gratopy.operator.opencl.OpenCLKernelSpec`.

All projection classes are concrete operator leaves. Their adjoints and
compositions retain those leaves and share their geometry and OpenCL runtime
caches.

Quick example
-------------

A Radon transform and its adjoint can be used as follows:

.. code-block:: python

    import numpy as np
    import pyopencl as cl
    import pyopencl.array as clarray
    import gratopy

    ctx = cl.create_some_context(interactive=False)
    queue = cl.CommandQueue(ctx)

    Nx = 128
    img = np.zeros((Nx, Nx), dtype=np.float32)

    R = gratopy.operator.Radon(image_domain=Nx, angles=180)

    sino = R.apply_to(img, queue=queue)
    backprojection = R.T.apply_to(sino)

A fan-beam transform additionally receives its source-to-detector and
source-to-origin distances:

.. code-block:: python

    F = gratopy.operator.Fanbeam(
        source_distances=(8.0, 4.0),
        image_domain=Nx,
        angles=360,
    )
    fan_sino = F.apply_to(img, queue=queue)
    fan_backprojection = F.T.apply_to(fan_sino)

The same operations can be written with operator syntax when the arrays
already reside on the OpenCL device:

.. code-block:: python

    device_img = clarray.to_device(queue, img)
    sino = R * device_img
    backprojection = R.T * sino

Queue selection is deterministic and does not depend on earlier applications.
An explicit ``queue=`` takes precedence; otherwise the queue is inferred from
a device argument or device output. Consequently, every application to a
NumPy array must either receive ``queue=`` explicitly or receive a
caller-provided :class:`pyopencl.array.Array` output. The multiplication
shorthand has no place to pass a queue and is therefore intended for device
arrays.

Output reuse and events
-----------------------

Applications allocate a result when ``output`` is omitted. Allocation-sensitive
iterative code can provide a compatible device output instead:

.. code-block:: python

    output = clarray.empty(queue, R.output_shape, dtype=np.float32)
    R.apply_to(device_img, output=output)

Compositions forward ``output`` to their final operation, while sums write
their first summand directly into it before accumulating the remaining terms.
Composite intermediates may still be allocated internally. Reusing outputs is
recommended in long iterative loops; retaining every newly returned output
necessarily retains the corresponding device memory.

OpenCL execution is asynchronous. Passing ``return_event=True`` returns
``(result, events)`` with the result's current event list in addition to
recording those events on the result array.

Detailed geometry example
-------------------------

The operator API also supports explicit geometry helper objects from
:mod:`gratopy.utilities`. This is often the clearest way to specify image
extent, detector geometry, shifts, and angular sampling explicitly.

.. code-block:: python

    import numpy as np
    import pyopencl as cl
    import gratopy
    from gratopy.utilities import Angles, Detectors, ImageDomain

    ctx = cl.create_some_context(interactive=False)
    queue = cl.CommandQueue(ctx)

    img = np.zeros((192, 128), dtype=np.float32)

    image_domain = ImageDomain(
        size=(192, 128),
        extent=3.0,
        center=(0.1, -0.2),
    )

    angles = Angles.uniform_interval(
        start=0.0,
        end=np.pi / 2,
        number=120,
    )

    detectors = Detectors(
        number=220,
        extent=3.0,
        center=0.15,
        reversed=False,
    )

    R = gratopy.operator.Radon(
        image_domain=image_domain,
        angles=angles,
        detectors=detectors,
    )

    sino = R.apply_to(img, queue=queue)
    backproj = R.T.apply_to(sino)

The same setup can also be written directly inline when constructing the
operator:

.. code-block:: python

    R = gratopy.operator.Radon(
        image_domain=ImageDomain(size=(192, 128), extent=3.0, center=(0.1, -0.2)),
        angles=Angles.uniform_interval(0.0, np.pi / 2, 120),
        detectors=Detectors(number=220, extent=3.0, center=0.15),
    )

This explicit style is particularly useful when experimenting with geometry in
Python code, because image domain, angles, and detector settings become
immutable first-class values that can be safely reused by multiple operators.
To change a detector or image setting, construct a new value, for example with
:func:`dataclasses.replace`. ``Angles`` makes private copies of its input arrays
and exposes them read-only so subsequent changes to caller-owned arrays cannot
invalidate an operator's cached geometry.

Extent placeholders
-------------------

For Radon operators, one physical extent can be inferred from the other by
using :class:`gratopy.utilities.ExtentPlaceholder`. This is useful when one
wants either the smallest detector covering a fixed image domain, or the
largest image domain covered by a fixed detector.

For example, the detector extent can be inferred from a fixed image extent:

.. code-block:: python

    from gratopy.utilities import Detectors, ExtentPlaceholder, ImageDomain

    R = gratopy.operator.Radon(
        image_domain=ImageDomain(size=128, extent=2.0),
        angles=180,
        detectors=Detectors(number=200, extent=ExtentPlaceholder.FULL),
    )

Conversely, the image extent can be inferred from a fixed detector extent:

.. code-block:: python

    R = gratopy.operator.Radon(
        image_domain=ImageDomain(size=128, extent=ExtentPlaceholder.FULL),
        angles=180,
        detectors=Detectors(number=200, extent=2.0),
    )

Only one side may use an extent placeholder at a time. Passing placeholders for
both the image and detector extents is unsupported and raises
:class:`NotImplementedError`. If the requested placeholder semantics are
geometrically impossible for the supplied centers and fixed extent, construction
raises :class:`ValueError`.

Adjoint convention
------------------

Projection adjoints use the same weighted discretization as the legacy API.
Angular quadrature weights from :class:`gratopy.utilities.Angles` are included
in each backprojection kernel. Thus ``R.T`` and ``F.T`` denote adjoints with
respect to gratopy's physical image and sinogram pairings; they are not
generally plain Euclidean transposes of the unweighted forward-projection
matrices.

Operator algebra
----------------

Operators inherit from :class:`gratopy.operator.base.Operator`, which builds
non-mutating expression nodes for arithmetic, scaling, adjoints, and
composition. Expression nodes retain references to their original operands;
they do not copy concrete operators or their runtime caches. For example, one
can form a Gram operator

.. code-block:: python

    G = R.T * R

and apply it to an image:

.. code-block:: python

    gram_img = G.apply_to(img, queue=queue)

This is one of the main motivations for the operator interface: projection
operators can be combined with a syntax that mirrors the underlying linear
algebra. The multiplication syntax is intentionally overloaded:

- ``A * B`` composes two operators,
- ``alpha * A`` and ``A * alpha`` scale an operator,
- ``A * x`` applies an operator to a non-operator argument.

Addition and subtraction construct sum expressions. Algebra creates dedicated
expression nodes and never mutates or copies concrete leaves.

Norm estimation
---------------

Every operator provides :meth:`gratopy.operator.base.Operator.norm_estimate`.
The default ``"poweriteration"`` algorithm applies power iteration to
``A.T * A`` and supports OpenCL operators via an explicit queue:

.. code-block:: python

    estimate = R.norm_estimate(queue=queue, number_iterations=30)

The alternative ``"naive"`` algorithm combines leaf values structurally using
the triangle inequality for sums and submultiplicativity of operator norms for
compositions. These inequalities preserve certified upper bounds, but unknown
leaf norms are currently obtained from finite power iterations and are not
themselves certified upper bounds. The resulting combined value is therefore a
heuristic unless certified bounds are available for every leaf.

Class structure
---------------

The operator implementation is intentionally layered in a small number of
classes.

As an **experimental** interface, the operator API may still change in
backward-incompatible ways without a full deprecation cycle while the design is
settling.

:class:`gratopy.operator.base.Operator`
    Provides the common operator interface and constructs dedicated adjoint,
    scale, sum, and composition expression nodes. Concrete operands are shared
    across the expression tree rather than copied.

:class:`gratopy.operator.opencl._OpenCLOperator`
    Internal helper base for OpenCL-backed operators. It implements shared
    execution plumbing such as queue inference, array coercion, output
    allocation, program cacheing, and kernel lookup.

:class:`gratopy.operator.projection.Radon` and :class:`gratopy.operator.projection.Fanbeam`
    Concrete projection operators. They own immutable geometry state,
    geometry-specific preparation, and bindings to the OpenCL kernels. The
    ray- and strip-driven classes are dedicated subclasses selecting alternate
    kernel bundles while preserving the same execution and cache lifecycle.

This separation keeps the generic algebra in :mod:`gratopy.operator.base`
backend-agnostic while concentrating OpenCL-specific behavior in a gratopy-
specific internal layer.

Compiled-program lifecycle
--------------------------

Compiled OpenCL programs are shared across concrete operators using the same
context, kernel source, build options, and template-expansion mode. The global
registry retains these program bundles weakly; each operator that has actually
used a bundle retains a strong runtime lease. Consequently, equivalent live
operators share compilation work, while a bundle is released automatically
after its last operator lease disappears. Adjoint and composite expressions
retain their concrete leaves and therefore participate in the same lifecycle.

Kernel instances are local to each calling thread because OpenCL kernel
arguments are mutable. Threads share the compiled program but not argument
state. This protects kernel argument setup, but it does not yet constitute a
guarantee that complete operators can be applied concurrently: other lazy
runtime caches still require a dedicated thread-safety pass.

The registry can be invalidated explicitly when required:

.. code-block:: python

    from gratopy.operator import invalidate_kernel_cache

    invalidate_kernel_cache()

Live operators acquire newly compiled bundles on their next application.
Invocations already in progress may finish with their existing program.
Invalidation removes bundles from future lookup; an existing operator may keep
its old lease alive until its next application or until the operator is
released.

Custom kernels
--------------

One goal of the new operator API is to make kernel experimentation easier.
Kernel sources can be specified via
:class:`gratopy.operator.opencl.OpenCLKernelSpec`.

A custom kernel spec can be passed directly to the operator:

.. code-block:: python

    from gratopy.operator import OpenCLKernelSpec

    spec = OpenCLKernelSpec.from_path("scratch/my_radon.cl", base_name="radon")
    R = gratopy.operator.Radon(image_domain=128, angles=180, kernel_spec=spec)

This allows experimenting with alternative kernels while keeping the Python-side
operator interface unchanged.

Subclassing `_OpenCLOperator`
-----------------------------

For experimental custom operators, the internal class
:class:`gratopy.operator.opencl._OpenCLOperator` provides a default
:py:meth:`apply_to() <gratopy.operator.opencl._OpenCLOperator.apply_to>`
implementation. Although `_OpenCLOperator` is internal, its documented hook
methods form the intended customization surface for OpenCL-backed operators.

The default execution pipeline performs, in order:

1. queue inference,
2. coercion of array-like inputs to :class:`pyopencl.array.Array`,
3. direction-aware input validation,
4. output allocation (if needed),
5. output validation,
6. forward or adjoint kernel lookup,
7. kernel invocation.

Scalar multiplication is represented by a dedicated expression node instead
of mutating or copying the concrete OpenCL operator.

Subclasses can adapt this behavior mostly via hooks instead of overriding the
entire method.

Important hooks are:

- :py:meth:`gratopy.operator.opencl._OpenCLOperator._default_kernel_spec`
  for the default kernel bundle,
- :py:meth:`gratopy.operator.opencl._OpenCLOperator._expected_output_shape`
  for shape inference,
- :py:meth:`gratopy.operator.opencl._OpenCLOperator._validate_argument`
  and
  :py:meth:`gratopy.operator.opencl._OpenCLOperator._validate_output`
  for validation,
- :py:meth:`gratopy.operator.opencl._OpenCLOperator._get_kernel`
  for choosing the compiled kernel,
- :py:meth:`gratopy.operator.opencl._OpenCLOperator._kernel_arguments`
  for supplying additional kernel arguments beyond output and input buffers,
- :py:meth:`gratopy.operator.opencl._OpenCLOperator._global_shape`
  for customizing the OpenCL launch shape.

In simple cases, a custom operator only needs to provide a kernel spec and
static input/output shapes. The shared OpenCL implementation dispatches the
forward and adjoint kernels through :meth:`apply_to` and
:meth:`apply_adjoint_to`; :attr:`T` is an adjoint expression wrapper around the
same concrete operator and therefore shares its runtime caches.

Limitations and status
----------------------

The operator API is still evolving. In particular:

- extent placeholders are currently supported experimentally for parallel-beam
  operators when exactly one of the image or detector extents is a placeholder;
  fan-beam operators require numeric extents,
- higher-level solver interfaces are still centered around the legacy API,
- the alternative ray- and strip-driven kernels are experimental and may still
  evolve as their numerical behavior is characterized across more devices.

For the full and mature feature set of gratopy, the legacy API documented in
:doc:`getting_started` and :doc:`functions` remains the main reference.
