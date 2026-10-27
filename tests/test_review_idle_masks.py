import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location('review_idle', Path(__file__).parents[1] / 'scripts/export_review_idle_masks.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_generated_waits_keep_departure_and_terminal_arrival_supervision():
    clock = [0, 0, 0, 1, 2, 2, 2, 3, 4, 4, 4]
    expected = [1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1]
    np.testing.assert_array_equal(module.clock_idle_mask(clock), expected)
    np.testing.assert_array_equal(np.asarray(clock)[~module.clock_idle_mask(clock)], [0, 1, 2, 3, 4])


def test_slow_ramps_and_original_source_progress_are_not_idle():
    assert not module.clock_idle_mask([0, 1e-12, 0.01, 0.1, 1, 2]).any()
    assert not module.clock_idle_mask([4]).any()
    assert module.clock_idle_mask([4, 4, 4]).all()


@pytest.mark.parametrize('clock', [[], [1, 0], [0, np.nan], [[0, 1]]])
def test_invalid_clocks_are_rejected(clock):
    with pytest.raises(ValueError):
        module.clock_idle_mask(clock)


def test_left_wait_switches_to_idle_at_opening_and_keeps_departure():
    left = np.array([0, 1, 2, 2, 2, 2, 2, 3, 4])
    right = np.arange(9)
    stages = dict(peak_source_frame=2, open_source_frame=4, right_ready_source_frame=6)
    required, excess = module.left_wait_masks(left, right, stages)
    np.testing.assert_array_equal(np.flatnonzero(required), [2, 3])
    np.testing.assert_array_equal(np.flatnonzero(excess), [4, 5])
    assert not excess[6:].any()  # Departure and motion stay supervised.


def test_open_drawer_does_not_mask_lifting_or_uninterrupted_insertion():
    stages = dict(peak_source_frame=3, open_source_frame=1, right_ready_source_frame=5)
    required, excess = module.left_wait_masks(np.arange(7), np.arange(7), stages)
    assert not required.any() and not excess.any()
    required, excess = module.left_wait_masks([0,1,3,3,3,4,5], np.arange(7), stages)
    np.testing.assert_array_equal(np.flatnonzero(excess), [2,3])


def test_continuous_close_wait_stops_before_preparation():
    left = np.arange(10)
    right = np.arange(10)
    stages = dict(open_source_frame=1, withdrawal_source_frame=3, close_source_frame=9)
    required, excess = module.right_wait_masks(left, right, stages, {'close_preparation_source_frame': 7})
    np.testing.assert_array_equal(np.flatnonzero(required), [1, 2])
    np.testing.assert_array_equal(np.flatnonzero(excess), [3, 4, 5, 6])
    assert not excess[7:].any()
    _, no_wait = module.right_wait_masks(left, right, stages, {'close_preparation_source_frame': 2})
    assert not no_wait.any()


@pytest.mark.parametrize('boundary', [0, 10, 3.5, np.nan])
def test_invalid_close_preparation_rejected(boundary):
    with pytest.raises(ValueError):
        module.right_wait_masks(np.arange(10), np.arange(10),
            dict(open_source_frame=1, withdrawal_source_frame=3, close_source_frame=9),
            {'close_preparation_source_frame': boundary})


def test_export_requires_complete_phase_coverage(tmp_path):
    import json
    manifest = tmp_path / 'manifest.json'
    boundaries = tmp_path / 'boundaries.json'
    manifest.write_text(json.dumps({'episodes_data': [{'source': 3}]}))
    boundaries.write_text(json.dumps({'sources': {}}))
    with pytest.raises(ValueError, match='exactly every source'):
        module.export(tmp_path, manifest, tmp_path / 'out.json', boundaries)


def test_terminal_rest_starts_after_return_not_interior_wait():
    state = np.zeros((120, 7))
    state[30:60, 0] = np.arange(30)
    state[60:, 0] = 29
    result = module.terminal_rest(state, state, 10)
    assert result == dict(start=59, end=120)
    # Other arm returns later and has no excess tail.
    later = np.zeros((120, 7))
    later[:90, 0] = np.arange(90)
    later[90:, 0] = 89
    assert module.terminal_rest(later, later, 10)['start'] == 89


def test_terminal_rest_checks_commands_gripper_and_accumulated_drift():
    state = np.zeros((100, 7)); action = state.copy()
    action[:80, 6] = np.arange(80)
    action[80:, 6] = 79
    assert module.terminal_rest(state, action, 10)['start'] == 79
    drift = np.zeros((100, 7)); drift[:, 0] = np.arange(100) * .1
    assert module.terminal_rest(drift, drift, 0)['start'] >= 96
    assert module.terminal_rest(state, state, 70)['start'] == 70


def test_reviewed_retreat_advances_only_right_idle_labels():
    left=right=np.arange(10)
    stages=dict(open_source_frame=1,withdrawal_source_frame=6,close_source_frame=9)
    boundary=dict(close_preparation_source_frame=8,mask_withdrawal_source_frame=3)
    required,excess=module.right_wait_masks(left,right,stages,boundary)
    np.testing.assert_array_equal(np.flatnonzero(required),[1,2])
    np.testing.assert_array_equal(np.flatnonzero(excess),[3,4,5,6,7])
    assert stages['withdrawal_source_frame']==6


def test_drawer_export_preserves_required_waits_and_excess_waits(tmp_path):
    import json
    import pyarrow as pa
    import pyarrow.parquet as pq

    for folder in ['data/chunk-000', 'meta/retime_receipts', 'meta/retime_source_indices']:
        (tmp_path/folder).mkdir(parents=True)
    left = np.array([0,1,2,2,2,2,3,4,5,6,7,7.0])
    right = np.arange(12.0)
    state = np.column_stack([np.repeat(left[:,None],7,axis=1),np.repeat(right[:,None],7,axis=1)])
    table = pa.table({'retime.left_source_frame': left, 'retime.right_source_frame': right,
                      'retime.synthetic_hold': np.zeros(12,bool),
                      'action':state.tolist(), 'observation.state':state.tolist()})
    pq.write_table(table,tmp_path/'data/chunk-000/file-000.parquet')
    np.savez(tmp_path/'meta/retime_source_indices/episode_000.npz',left=left,right=right)
    stages=dict(peak_source_frame=2,open_source_frame=4,withdrawal_source_frame=6,close_source_frame=10)
    (tmp_path/'meta/retime_receipts/episode_000.json').write_text(json.dumps({'source_episode_index':3,'plan':{'stages':stages}}))
    manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps({'fps':30,'episodes_data':[{'output':0,'source':3,'frames':12}]}))
    boundaries=tmp_path/'boundaries.json';boundaries.write_text(json.dumps({'sources':{'3':{'close_preparation_source_frame':10}}}))
    result=module.export(tmp_path,manifest,tmp_path/'masks.json',boundaries)
    row=result['episodes'][0]
    expected=np.ones((12,2),np.float32);expected[4,0]=0;expected[9,1]=0
    np.testing.assert_array_equal(row['observation.arm_active_mask'],expected)
    assert row['left']==[[4,5]] and row['right']==[[9,10]]
