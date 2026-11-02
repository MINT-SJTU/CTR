import cv2
import numpy as np

from real_robot_data_retime.water_segmentation import clean_masks


def test_return_cleanup_removes_tray_but_keeps_attached_dark_arm():
    image = np.full((240, 320, 3), 230, np.uint8)
    hsv = np.uint8([[[20, 150, 160]]])
    wood = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    image[60:200, 90:230] = wood
    frames = np.repeat(image[None], 3, axis=0)
    frames[0, 100:115, 20:130] = 20
    masks = np.zeros((3, 2, 240, 320), bool)
    masks[0, 0, 60:200, 90:230] = True  # Observed tracker latch.
    masks[0, 0, 100:115, 20:130] = True
    clean_masks(masks, frames, "post")
    assert not masks[0, 0, 140:180, 140:200].any()
    assert masks[0, 0, 105:110, 25:125].all()


def test_boundary_supplement_preserves_existing_other_arm_identity():
    from real_robot_data_retime.water_segmentation import add_entry_fragments

    frame = np.full((80, 640, 3), 255, np.uint8)
    frame[60:80, :20] = 0
    robots = np.zeros((2, 80, 640), bool)
    robots[1, 65:75, 5:15] = True
    original_right = robots[1].copy()
    add_entry_fragments(robots, frame)
    np.testing.assert_array_equal(robots[1], original_right)
    assert not (robots[0] & robots[1]).any()
    assert robots[0, 60:65, :20].all()


def test_boundary_supplement_does_not_hide_preexisting_conflicts():
    import pytest
    from real_robot_data_retime.water_segmentation import add_entry_fragments

    robots = np.ones((2, 80, 640), bool)
    with pytest.raises(ValueError, match="ambiguous"):
        add_entry_fragments(robots, np.zeros((80, 640, 3), np.uint8))


def test_observed_wide_background_rail_excluded_but_visible_robot_kept():
    from real_robot_data_retime.water_segmentation import exclude_observed_rail
    background = np.full((120, 180, 3), 230, np.uint8)
    background[20:27] = 60
    frames = np.repeat(background[None], 3, axis=0)
    frames[1, 18:40, 25:50] = 5  # Visible articulated foreground over the rail.
    masks = np.zeros((3, 2, 120, 180), bool)
    masks[:, 0, 20:27] = True  # Erroneous whole-table appendage.
    masks[1, 0, 18:40, 25:50] = True
    audit = exclude_observed_rail(masks, frames, background, 80)
    assert audit['removed_pixels'] > 0
    assert not masks[0].any()
    assert masks[1, 0, 18:40, 25:50].all()
    assert not masks[1, 0, 20:27, 80:150].any()


def test_static_rail_one_pixel_camera_shift_is_not_a_robot_fragment():
    from real_robot_data_retime.water_segmentation import exclude_observed_rail
    background=np.full((120,180,3),230,np.uint8);background[20:27]=60
    frame=np.roll(background,1,axis=0);frame[30:45,20:40]=5
    masks=np.zeros((1,2,120,180),bool);masks[0,0,21:28]=True;masks[0,0,30:45,20:40]=True
    exclude_observed_rail(masks,frame[None],background,80)
    assert not masks[0,0,21:28,80:150].any()
    assert masks[0,0,30:45,20:40].all()
