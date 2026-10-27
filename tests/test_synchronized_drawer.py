import numpy as np
from real_robot_data_retime.timeline.synchronized_drawer import right_entry_profile,entry_edge


def test_ready_arm_waits_at_entry_not_past_it():
    assert entry_edge(8,10,9,10,10,20)
    assert not entry_edge(8,10,9,11,10,20)
    assert entry_edge(10,9,10,10,10,20)
    assert entry_edge(10,10,11,11,10,20)
    assert not entry_edge(10,9,11,10,10,20)
    assert not entry_edge(11,11,12,11,10,20)
    assert not entry_edge(11,11,13,13,10,20)


def test_common_phase_restores_skipped_frames_and_preserves_exterior():
    right=np.array([0,1,5,10,15,20,25])
    left=np.arange(5,26,dtype=float)
    result,entry=right_entry_profile(right,left,5,20)
    assert entry==3
    np.testing.assert_array_equal(result[:entry],[0,1,5])
    np.testing.assert_array_equal(result[entry:-1],np.arange(10,21))
    assert result[-1]==25


def test_common_phase_rejects_independent_wrist_clock_or_skips():
    import pytest
    from real_robot_data_retime.timeline.synchronized_drawer import validate_common_phase
    phase=dict(start_source_frame=2,end_source_frame=4)
    assert validate_common_phase([0,1,2,3,4,5],[0,2,2,3,4,6],phase)['passed']
    with pytest.raises(ValueError):validate_common_phase([0,1,2,3,4,5],[0,2,2,3.2,4,6],phase)
    with pytest.raises(ValueError):validate_common_phase([0,2,4],[0,2,4],phase)
