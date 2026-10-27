"""Schedule right-arm readiness before replaying the common interaction."""
import numpy as np


def right_entry_profile(right_sources, left_clock, peak_index, end_source):
    """Preserve prefix/suffix motions and share the left phase's clock samples."""
    start = left_clock[peak_index]
    end_index = int(np.searchsorted(left_clock, end_source))
    if end_index >= len(left_clock) or left_clock[end_index] != end_source:
        raise ValueError('Common interaction end is absent from left clock')
    prefix = right_sources[right_sources < start]
    shared = left_clock[peak_index:end_index + 1]
    suffix = right_sources[right_sources > end_source]
    return np.r_[prefix, shared, suffix], len(prefix)


def entry_edge(left, right, next_left, next_right, start, end):
    """Allow waiting at entry; prohibit any independent movement inside."""
    if left < start and next_left > start:
        return False
    if left <= start and right > start:
        return False
    if next_left <= start and next_right > start:
        return False
    if start < next_left <= end or start <= left < end and next_left > start:
        return left >= start and left == right and next_left == next_right and 0 < next_left - left <= 1
    return True


def validate_common_phase(left, right, phase):
    left, right = np.asarray(left), np.asarray(right)
    start, end = phase['start_source_frame'], phase['end_source_frame']
    begins = np.flatnonzero((left == start) & (right == start))
    ends = np.flatnonzero((left == end) & (right == end))
    if not len(begins) or not len(ends):
        raise ValueError('Common interaction boundaries are not aligned')
    a, b = int(begins[-1]), int(ends[0])
    if b <= a or not np.array_equal(left[a:b+1], right[a:b+1]):
        raise ValueError('Common interaction source clocks differ')
    delta = np.diff(left[a:b+1])
    if np.any(delta <= 0) or np.any(delta > 1):
        raise ValueError('Common interaction skips or repeats source motion')
    return dict(start_output_frame=a, end_output_frame=b, passed=True)
