import numpy as np
import real_robot_data_retime.interaction.contact_grippers as module


def inputs():
    frames = np.zeros((20, 100, 200, 3), np.uint8)
    mask = np.zeros((100, 200), bool)
    mask[70:80, 90:100] = True
    robots = np.zeros((20, 2, 100, 200), bool)
    robots[:, 0, 45:85, 65:115] = True
    grippers = dict(
        centers=np.full((20, 2, 2), np.nan),
        apertures=np.ones((20, 2)),
        masks=robots.copy(),
        robot_masks=np.packbits(robots, axis=-1),
    )
    return frames, [dict(mask=mask, origin=np.array([95.0, 75.0]))], grippers


class UnexpectedModel:
    def propagate(self, *args, **kwargs):
        raise AssertionError("no independent contact: must not segment")


def test_no_contact_leaves_measurements_unchanged(monkeypatch):
    monkeypatch.setattr(module, "origin_departure_interval", lambda *args: (None, None))
    frames, proposals, grippers = inputs()
    result, audit = module.recover_contact_grippers(
        frames, proposals, grippers, [0], UnexpectedModel(), 10
    )
    np.testing.assert_array_equal(result["centers"], grippers["centers"])
    assert result["robot_masks"] is grippers["robot_masks"]
    assert audit[0]["status"] == "no_observed_contact"


def test_ambiguous_arm_contact_is_not_assigned(monkeypatch):
    monkeypatch.setattr(
        module,
        "origin_departure_interval",
        lambda *args: (
            {"last_observed_at_origin": 6, "first_observed_empty": 15},
            None,
        ),
    )
    frames, proposals, grippers = inputs()
    grippers["robot_masks"][:, 1] = grippers["robot_masks"][:, 0]
    _, audit = module.recover_contact_grippers(
        frames, proposals, grippers, [0], UnexpectedModel(), 10
    )
    assert audit[0]["status"] == "ambiguous_contact_arm"


def test_masks_unrelated_to_the_robot_cannot_replace_gripper(monkeypatch):
    monkeypatch.setattr(
        module,
        "origin_departure_interval",
        lambda *args: (
            {"last_observed_at_origin": 6, "first_observed_empty": 15},
            None,
        ),
    )

    class Model:
        def propagate(self, frames, proposals, seed_frame, **kwargs):
            mask = np.zeros((1, 100, 200), bool)
            mask[:, 10:30, 150:170] = True
            yield seed_frame, mask

    frames, proposals, grippers = inputs()
    result, audit = module.recover_contact_grippers(
        frames, proposals, grippers, [0], Model(), 10
    )
    np.testing.assert_array_equal(result["centers"], grippers["centers"])
    assert audit[0]["measured_frames"] == []
