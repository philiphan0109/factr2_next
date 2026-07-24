import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")

from factr2_next.data_collection.h5_writer import H5Writer
from factr2_next.evaluation.plot_episode import plot_episode
from factr2_next.evaluation.recorder_node import ALL_KEYS, FreshSamples, SCALAR_KEYS, VECTOR_KEYS


def _raw():
    return {
        "joint_pos": np.zeros(8),
        "joint_vel": np.zeros(8),
        "joint_cmd": np.zeros(8),
        "measured_joint_torque": np.zeros(7),
    }


def _cache(received=1.0):
    cache = FreshSamples(0.1)
    for key in VECTOR_KEYS:
        cache.update(key, np.zeros(7), received)
    for key in SCALAR_KEYS:
        cache.update(key, [1.0], received)
    return cache


def test_row_shapes_scalars_and_stale_handling():
    cache = _cache()
    row, missing = cache.assemble(_raw(), now=1.05)
    assert not missing
    assert row["joint_pos"].shape == (8,)
    assert row["feedback_torque"].shape == (7,)
    assert row["score"].shape == (1,)
    row, missing = cache.assemble(_raw(), now=1.2)
    assert row is None
    assert set(missing) == set(VECTOR_KEYS + SCALAR_KEYS)


def test_plot_synthetic_episode(tmp_path):
    h5_path = tmp_path / "eval.h5"
    writer = H5Writer(h5_path, "test", ALL_KEYS,
                      {"contact_low_threshold": 1.0, "contact_high_threshold": 1.5})
    writer.start_episode()
    cache = _cache()
    row, _ = cache.assemble(_raw(), now=1.0)
    for i in range(12):
        sample = {key: value.copy() for key, value in row.items()}
        sample["external_joint_torque_raw"][:] = np.sin(i / 3.0)
        sample["external_joint_torque_filtered"][:] = np.sin(i / 4.0)
        sample["score"][:] = i / 5.0
        sample["contact_state"][:] = i > 5
        writer.append(1_000_000_000 + i * 16_666_667, sample)
    writer.close()
    output = plot_episode(h5_path, output=tmp_path / "figure.png", ema_alpha=0.2)
    assert output.stat().st_size > 0
    with h5py.File(h5_path, "r") as h5:
        ep = h5["ep_0000"]
        assert {ep[key]["data"].shape[0] for key in ALL_KEYS} == {12}
        assert ep["score"]["data"].shape == (12, 1)
