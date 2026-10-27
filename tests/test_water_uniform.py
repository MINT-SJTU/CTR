import numpy as np
from real_robot_data_retime.timeline.water_uniform import grid_indices, make_uniform_schedule

def test_even_half_covers_all_sources_without_changing_shared_or_terminal_supervision():
    selected = []
    for source in range(75):
        kept = 0
        for variant, grid in enumerate(grid_indices(source)):
            if grid % 2:
                continue
            kept += 1
            selected.append(grid)
            s = make_uniform_schedule(700, 220, 590, source, variant, prefix=4)
            c, a = s['clocks'], s['active']
            np.testing.assert_array_equal(c[s['shared'], 0], c[s['shared'], 1])
            assert (a[s['shared']] == 1).all()
            assert (a[-45:] == 1).all()
            assert (c[-45:] == 699).all()
            assert not (a == 0).all(axis=1).any()
            moving = np.diff(c, axis=0) > 0
            assert (a[1:][moving] == 1).all()
            assert (a[:-1][moving] == 1).all()
            assert (np.diff(c, axis=0) <= 1).all()
        assert kept == 2
    assert sorted(selected) == list(range(0, 300, 2))
