import copy
import pickle
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from gratopy.utilities import Angles, Detectors, ExtentPlaceholder, ImageDomain


def test_angles_are_immutable_and_own_read_only_array_copies():
    angle_values = np.array([0.0, 1.0])
    weight_values = np.array([0.5, 0.5])
    angles = Angles(angle_values, weight_values, half_circle=True)

    angle_values[0] = 2.0
    weight_values[0] = 3.0

    np.testing.assert_array_equal(angles.angles, [0.0, 1.0])
    np.testing.assert_array_equal(angles.weights, [0.5, 0.5])
    assert not angles.angles.flags.writeable
    assert not angles.weights.flags.writeable

    with pytest.raises(ValueError, match="read-only"):
        angles.angles[0] = 2.0
    with pytest.raises(ValueError, match="read-only"):
        angles.weights[0] = 3.0
    with pytest.raises(ValueError, match="WRITEABLE"):
        angles.angles.flags.writeable = True
    with pytest.raises(FrozenInstanceError):
        angles.half_circle = False


def test_angles_remain_immutable_after_copying_and_pickling():
    angles = Angles([0.0, 1.0], [0.5, 0.5])

    restored_angles = (
        copy.deepcopy(angles),
        pickle.loads(pickle.dumps(angles)),
    )

    for restored in restored_angles:
        np.testing.assert_array_equal(restored.angles, angles.angles)
        assert not restored.angles.flags.writeable
        with pytest.raises(ValueError, match="WRITEABLE"):
            restored.angles.flags.writeable = True


def test_detectors_are_immutable():
    detectors = Detectors(number=-20, extent=3.0, center=0.25)

    assert detectors.number == 20
    assert detectors.reversed is True
    with pytest.raises(FrozenInstanceError):
        detectors.extent = 4.0


def test_image_domains_are_immutable():
    image_domain = ImageDomain(
        size=(16, 12),
        extent=2.0,
        center=(0.1, -0.2),
    )

    assert image_domain.size == (16, 12)
    assert image_domain.center == (0.1, -0.2)
    with pytest.raises(FrozenInstanceError):
        image_domain.extent = ExtentPlaceholder.FULL
