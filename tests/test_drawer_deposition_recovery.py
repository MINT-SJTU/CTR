import cv2
import numpy as np
import pytest
from real_robot_data_retime.interaction.drawer_recovery import deposition_observations


def scene():
    frames=np.full((12,100,160,3),220,np.uint8)
    blue=cv2.cvtColor(np.uint8([[[85,200,200]]]),cv2.COLOR_HSV2BGR)[0,0]
    frames[:,25:34,112:121]=blue  # A static same-colored cabinet distractor.
    frames[3:,50:59,95:104]=blue  # New deposited object.
    frames[3:,72:82,125:135]=blue  # A robot-attached same-colored patch.
    robots=np.zeros((12,2,100,160),bool);robots[3:,0,72:82,125:135]=True
    interior=np.zeros((100,160),bool);interior[20:90,80:145]=True
    proposal=dict(color=np.array([85,200,200]),area=81,origin=np.array([20.,80.]))
    return frames,interior,proposal,robots


def test_deposition_seed_excludes_static_color_and_robot_attached_distractors():
    f,m,p,r=scene();seed,audit=deposition_observations(f,m,p,r,0,12,10,after=2)
    np.testing.assert_allclose(seed['center'],[99,54])
    assert audit['visible_deposition_frames']==list(range(3,12))


def test_no_deposition_is_invented_from_static_and_attached_blue_pixels():
    f,m,p,r=scene();f[:,50:59,95:104]=220
    with pytest.raises(ValueError,match='no persistent'):
        deposition_observations(f,m,p,r,0,12,10,after=2)
