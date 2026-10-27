import numpy as np
from real_robot_data_retime.interaction.verification import (
    continuous_workspace_identity,
)


def fixture():
    x = np.r_[np.full(10, 150.0), np.linspace(150, 200, 40), np.full(10, 200.0)]
    xy = np.column_stack([x, np.full(60, 150.0)])
    return [{"centers": xy}], [
        {
            "origin": np.array([150.0, 150.0]),
            "bbox": [140, 140, 20, 20],
            "appearance_model": "darkness",
        }
    ]


def test_nudged_object_retains_continuous_identity():
    tracks, proposals = fixture()
    result = continuous_workspace_identity(tracks, proposals, 0, 60, 30, (240, 400))
    assert result["verified"]
    assert result["pregrasp_center"][0] > 190


def test_identity_cannot_cross_long_missing_interval_or_jump():
    tracks, proposals = fixture()
    tracks[0]["centers"][20:40] = np.nan
    assert not continuous_workspace_identity(tracks, proposals, 0, 60, 30, (240, 400))[
        "verified"
    ]
    tracks, proposals = fixture()
    tracks[0]["centers"][30] = [300, 150]
    assert not continuous_workspace_identity(tracks, proposals, 0, 60, 30, (240, 400))[
        "verified"
    ]


def test_deposited_or_duplicated_object_is_not_new_workspace_grasp():
    tracks, proposals = fixture()
    tracks[0]["centers"][-10:, 0] = 40
    assert not continuous_workspace_identity(tracks, proposals, 0, 60, 30, (240, 400))[
        "verified"
    ]
    tracks, proposals = fixture()
    tracks.append({"centers": tracks[0]["centers"].copy()})
    proposals.append(dict(proposals[0]))
    assert not continuous_workspace_identity(tracks, proposals, 0, 60, 30, (240, 400))[
        "verified"
    ]
