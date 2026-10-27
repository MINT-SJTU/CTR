import numpy as np
import pytest
from real_robot_data_retime.timeline.water import make_schedule,validate_schedule,ORDERS
@pytest.mark.parametrize('start',ORDERS)
@pytest.mark.parametrize('ret',ORDERS)
def test_all_nine_schedules_protect_tray_and_completion(start,ret):
 s=make_schedule(663,327,550,start,ret);c=s['clocks'];a=s['active'];np.testing.assert_array_equal(c[s['shared'],0],c[s['shared'],1]);assert np.all(a[-45:]==1);assert np.all(c[-45:]==662);assert np.all(a[s['required_wait']]==1)
 for side in range(2):
  changes=np.flatnonzero(np.diff(c[:,side])>0)+1;assert np.all(a[changes,side]==1);assert np.all(a[changes-1,side]==1)
 if start=='parallel' and ret=='parallel':assert np.all(a==1)
def test_corrupt_sync_clock_is_rejected():
 s=make_schedule(663,327,550,'parallel','parallel');s['clocks'][400,0]+=.1
 with pytest.raises(ValueError,match='diverge|skipped'):validate_schedule(s['clocks'],s['active'],s['shared'],s['required_wait'],s['synthetic'],663,327,550)
def test_terminal_idle_is_rejected():
 s=make_schedule(663,327,550,'left_first','right_first');s['active'][-1]=0
 with pytest.raises(ValueError,match='completion'):validate_schedule(s['clocks'],s['active'],s['shared'],s['required_wait'],s['synthetic'],663,327,550)
