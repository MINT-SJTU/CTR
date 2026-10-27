PROFILE = dict(
    name="workpiece",
    camera="observation.images.right_environment_1",
    expected_objects=4,
    object_kind="dark",
    dependencies=[],
    priority="alternating_pickups",
)


def discover_bins(frame):
    import cv2
    from ..interaction.video import components

    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    wood = (
        (hsv[:, :, 0] > 5)
        & (hsv[:, :, 0] < 35)
        & (hsv[:, :, 1] > 50)
        & (hsv[:, :, 2] > 65)
    )
    wood[: int(h * 0.35)] = False
    bins = [
        dict(bbox=stat[:4].tolist(), mask=mask, center=center)
        for mask, stat, center in components(wood, int(h * w * 0.01))
        if stat[2] > w * 0.15 and (center[0] < w * 0.25 or center[0] > w * 0.75)
    ]
    bins = sorted(bins, key=lambda b: b["bbox"][2] * b["bbox"][3], reverse=True)[:2]
    return sorted(bins, key=lambda b: b["center"][0])


def bin_visits(gripper, pickup, stop, bin_box, fps, *, robot_coverage):
    import numpy as np
    from scipy.ndimage import median_filter
    from ..interaction.evidence import stable_runs

    x, y, w, h = bin_box
    pad = max(w, h) * 0.25
    xy = np.asarray(gripper)
    finite = np.isfinite(xy).all(axis=1)
    near = (
        finite
        & (xy[:, 0] > x - pad)
        & (xy[:, 0] < x + w + pad)
        & (xy[:, 1] > y - pad)
        & (xy[:, 1] < y + h + pad)
    )
    coverage = np.asarray(robot_coverage, float)
    if coverage.shape != (len(xy),) or not np.isfinite(coverage).all():
        raise ValueError("invalid observed bin coverage")
    covered = median_filter((coverage >= 0.1).astype(np.uint8), size=5).astype(bool)
    velocity = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    quiet = (
        median_filter(np.where(np.isfinite(velocity), velocity, np.inf), size=5)
        < max(w, h) * 0.015
    )
    # Keep direct fingertip visits separate from arm-overlap hypotheses. Their
    # union could incorrectly merge two trips when the forearm spans the bin.
    for method, presence in [
        ("gripper_center", near),
        ("observed_arm_overlap", covered),
    ]:
        observed = presence.copy()
        observed[: pickup + 3] = False
        observed[stop:] = False
        for start, end in stable_runs(observed, max(3, round(fps * 0.15))):
            exits = np.zeros(stop - end, bool)
            if method == "gripper_center":
                exits = finite[end:stop] & ~near[end:stop]
            if np.max(coverage[start:end]) >= 0.1:
                clear_runs = stable_runs(coverage[end:stop] < 0.1, 3)
                if clear_runs:
                    exits[clear_runs[0][0]] = True
            outside = np.flatnonzero(exits) + end
            if not len(outside):
                continue
            clear_frame = int(outside[0])
            pauses = stable_runs(quiet[start : end - 1], 3)
            release = start + pauses[-1][1] if pauses else end - 1
            if method == "observed_arm_overlap":
                # With an unobserved tip, use the last frame before confirmed
                # clearance as a conservative upper bound on release.
                release = clear_frame - 1
            yield dict(
                entry_frame=start,
                release_frame=release,
                clearance_frame=clear_frame,
                release_interval=[start, clear_frame],
                visit_evidence=method,
            )


def verify_deposit(frames, robots, interaction_start, visit, bin_box, fps):
    import cv2
    import numpy as np

    x, y, w, h = map(int, bin_box)
    inset = max(2, round(min(w, h) * 0.03))
    roi = np.zeros(frames.shape[1:3], bool)
    roi[y + inset : y + h - inset, x + inset : x + w - inset] = True
    coverage = np.mean(robots[:, roi], axis=1)
    before = np.flatnonzero(coverage[:interaction_start] < 0.1)
    from ..interaction.evidence import stable_runs

    # Evidence must belong to this visit, before the next arm occlusion. Skipping
    # across another visit could credit a later object's deposit to this one.
    clear = stable_runs(coverage[visit["clearance_frame"] :] < 0.1, 1)
    if not len(before) or not clear:
        return dict(verified=False, reason="bin_not_observed_clear")
    start, stop = clear[0]
    start += visit["clearance_frame"]
    stop += visit["clearance_frame"]
    reference = int(before[-1])
    samples = np.arange(start, stop, max(1, round(fps * 0.2)))[:5]
    if len(samples) < 3:
        return dict(verified=False, reason="insufficient_post_deposit_evidence")
    first = frames[reference].astype(float)
    changed = []
    for t in samples:
        current = frames[t].astype(float)
        correction = np.median(current[roi] - first[roi], axis=0)
        darkening = np.mean(first + correction - current, axis=-1)
        change = (darkening > 20) & roi & ~robots[t]
        change = cv2.morphologyEx(
            change.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)
        ).astype(bool)
        changed.append(change)
    persistent = np.sum(changed, axis=0) >= max(3, len(changed) - 1)
    pixels = int(persistent.sum())
    return dict(
        verified=pixels >= 12,
        changed_pixels=pixels,
        reference_frame=reference,
        observed_frames=samples.tolist(),
        method="persistent_bin_appearance_change",
        reason="destination_changed"
        if pixels >= 12
        else "no_persistent_destination_change",
    )


def workpiece_events(events):
    """Validate and order the two source manipulations belonging to each arm."""
    own = [
        sorted(
            (e for e in events if e["robot_id"] == side),
            key=lambda e: e["pickup_frame"],
        )
        for side in ("left", "right")
    ]
    if [len(group) for group in own] != [2, 2]:
        raise ValueError("workpiece scheduling requires two pickups per arm")
    return own


def alternating_pickups(events):
    """Return the four source pickup milestones in shared-workspace order."""
    own = workpiece_events(events)
    return [
        (side, own[side][cycle]["pickup_frame"])
        for cycle in range(2)
        for side in range(2)
    ]


def pickup_precedence(left, right, milestones):
    clocks = (left, right)
    return all(
        clocks[side] < frame or clocks[previous_side] > previous_frame
        for (previous_side, previous_frame), (side, frame) in zip(
            milestones, milestones[1:]
        )
    )


def dark_object_masks(value, minimum_area, maximum_area):
    """Separate dark objects joined only by a brighter shadow saddle.

    Oversized components require independently large dark cores. Compact
    components are reconsidered only when the expected scene count is missing,
    and require substantial balanced cores. Partitioning preserves source pixels.
    """
    import numpy as np
    from scipy.ndimage import distance_transform_edt

    from ..interaction.video import components

    h, w = value.shape

    def partition(region, cores):
        labels = np.zeros((h, w), np.int32)
        for index, core in enumerate(cores, 1):
            labels[core] = index
        nearest = distance_transform_edt(
            labels == 0, return_distances=False, return_indices=True
        )
        owners = labels[tuple(nearest)]
        return [region & (owners == i) for i in range(1, len(cores) + 1)]

    mask = value < 95
    divided = []
    for region, stat, center in components(mask, minimum_area, maximum_area):
        if not (
            w * 0.2 < center[0] < w * 0.8
            and center[1] > h * 0.48
            and max(stat[2:4]) > w * 0.15
        ):
            continue
        for threshold in np.unique(value[region])[::-1]:
            cores = [
                core
                for core, bounds, _ in components(
                    region & (value < threshold), minimum_area, maximum_area
                )
                if max(bounds[2:4]) <= w * 0.15
            ]
            if len(cores) < 2:
                continue
            divided.extend(partition(region, cores))
            mask[region] = False
            break
    # A compact connected component can also contain two neighboring pieces.
    # Only reconsider these when the scene lacks the four expected objects;
    # require two substantial cores, not a bolt head split from its shaft.
    regions = []
    for support in [mask, *divided]:
        regions.extend(
            (region, stat, center)
            for region, stat, center in components(support, minimum_area, maximum_area)
            if w * 0.2 < center[0] < w * 0.8
            and center[1] > h * 0.48
            and max(stat[2:4]) <= w * 0.15
        )
    count = len(regions)
    for region, stat, _ in regions:
        if count >= PROFILE["expected_objects"]:
            break
        if not np.all(mask[region]):
            continue
        for threshold in np.unique(value[region])[::-1]:
            cores = [
                core
                for core, bounds, _ in components(
                    region & (value < threshold), minimum_area, maximum_area
                )
                if bounds[4] >= max(minimum_area, stat[4] * 0.25)
            ]
            if (
                len(cores) < 2
                or count + len(cores) - 1 > PROFILE["expected_objects"]
                or sum(core.sum() for core in cores) < stat[4] * 0.7
            ):
                continue
            divided.extend(partition(region, cores))
            mask[region] = False
            count += len(cores) - 1
            break
    return [mask, *divided]


def bin_aware_audit_support(frames, robots, motion, support):
    """Exclude independently observed stationary bin appearance from robot evidence."""
    import numpy as np
    from ..background.clean_plate import match_background_colors

    n, h, w = frames.shape[:3]
    bins = discover_bins(frames[0])
    if len(bins) != 2:
        raise ValueError("both destination bins must be observed for scene auditing")
    result = support.copy()
    motion = motion.copy()
    records = []
    for bin in bins:
        x, y, bw, bh = map(int, bin["bbox"])
        # Include the same three-pixel rim used by the scene compositor.
        x1, y1 = min(w, x + bw + 3), min(h, y + bh + 3)
        x, y = max(0, x - 3), max(0, y - 3)
        bw, bh = x1 - x, y1 - y
        bin_robots = np.array(
            [
                np.unpackbits(robots[t], axis=-1, count=w)[:, y:y1, x:x1].any(axis=0)
                for t in range(n)
            ]
        )
        clear = np.flatnonzero(bin_robots.mean(axis=(1, 2)) < 0.01)
        if len(clear) < 5:
            records.append(
                dict(bbox=[x, y, bw, bh], reference_frames=[], excluded_pixel_frames=0)
            )
            continue
        observed = clear[-5:]
        common = (~bin_robots & ~bin_robots[observed[-1]]).sum(axis=(1, 2))
        eligible = np.flatnonzero(common >= 100)
        indices = np.r_[eligible, observed[-1]]
        normalized, _ = match_background_colors(
            frames[indices, y:y1, x:x1], bin_robots[indices]
        )
        references = normalized[np.searchsorted(eligible, observed)].astype(float)
        reference = np.median(references, axis=0)
        stable = np.max(np.ptp(references, axis=0), axis=-1) <= 12
        excluded = 0
        for position, t in enumerate(eligible):
            stationary = stable & (
                np.max(np.abs(normalized[position].astype(float) - reference), axis=-1)
                <= 12
            )
            region = result[t, y:y1, x:x1]
            excluded += int((region & stationary).sum())
            region[stationary] = False
            motion[t, y:y1, x:x1][stationary] = 0
        records.append(
            dict(
                bbox=[x, y, bw, bh],
                reference_frames=observed.tolist(),
                excluded_pixel_frames=excluded,
            )
        )
    return motion, result, records


def refine_deposition_releases(timeline, state):
    """Bound verified bin releases by sustained opening of the measured jaws.

    Visual bin clearance can occur well into the return motion. A stable held
    aperture followed by at least 10 mm sustained opening inside that verified
    visit provides an earlier conservative release bound, without changing poses.
    """
    from copy import deepcopy
    import numpy as np

    state = np.asarray(state, float)
    if (
        state.ndim != 2
        or state.shape[1] != 14
        or not len(state)
        or not np.isfinite(state).all()
    ):
        raise ValueError("expected finite dual-arm measured states")
    refined = deepcopy(timeline)
    window = max(3, round(timeline["fps"] * 0.1))
    audit = []
    for event in refined["episodes"]:
        evidence = event.get("release_evidence")
        if not evidence or not evidence["verified"]:
            continue
        begin, end = evidence["entry_frame"], event["release_frame"]
        if begin < window or end >= len(state) or begin <= event["pickup_frame"]:
            continue
        aperture = state[:, 6 if event["robot_id"] == "left" else 13]
        held = aperture[begin - window : begin]
        if np.ptp(held) > 1.5:
            continue
        held_start = begin - window
        visit_end = min(end, evidence["clearance_frame"])
        # A broad visual bin visit can start before the physical grasp closes.
        # Require an observed >=10 mm closure before moving the held anchor.
        baseline = float(np.median(held))
        for candidate in range(begin, visit_end - 2 * window + 1):
            measured = aperture[candidate : candidate + window]
            if np.ptp(measured) <= 1.5 and measured.min() >= baseline + 10:
                break
            if np.ptp(measured) <= 1.5 and float(np.median(measured)) <= baseline - 10:
                held_start = candidate
                held = measured
                break
        minimum_open = float(np.median(held)) + 10.0
        for first in range(max(begin, held_start + window), visit_end - window + 1):
            opened = aperture[first : first + window]
            if opened.min() < minimum_open or np.ptp(opened) > 1.5:
                continue
            bound = first + window - 1
            item = {
                "arm": event["robot_id"],
                "object_id": event["object_id"],
                "visual_release_frame": end,
                "release_frame": bound,
                "held_frames": [held_start, held_start + window],
                "open_frames": [first, first + window],
                "held_aperture_mm": float(np.median(held)),
                "minimum_open_aperture_mm": float(opened.min()),
                "method": "verified_bin_visit_and_sustained_measured_jaw_opening",
            }
            event["release_frame"] = bound
            evidence["release_frame"] = bound
            evidence["measured_jaw_release"] = item
            audit.append(item)
            break
    return refined, audit


def verify_measured_releases(timeline, state, recorded):
    """Verify recorded jaw evidence, independent of which valid window was selected."""
    import numpy as np

    state = np.asarray(state, float)
    window = max(3, round(timeline["fps"] * 0.1))
    for item in recorded:
        event = next(
            e
            for e in timeline["episodes"]
            if e["robot_id"] == item["arm"] and e["object_id"] == item["object_id"]
        )
        evidence = event["release_evidence"]
        h0, h1 = item["held_frames"]
        o0, o1 = item["open_frames"]
        if not (
            evidence["verified"]
            and 0 <= h0 < h1 <= o0 < o1 <= len(state)
            and h1 - h0 == o1 - o0 == window
            and evidence["entry_frame"] <= o0
            and o1 - 1 <= min(item["visual_release_frame"], evidence["clearance_frame"])
            and o1 - 1 == item["release_frame"] == event["release_frame"]
        ):
            raise ValueError("invalid measured release evidence interval")
        jaw = state[:, 6 if item["arm"] == "left" else 13]
        held, opened = jaw[h0:h1], jaw[o0:o1]
        if not (
            np.ptp(held) <= 1.5
            and np.ptp(opened) <= 1.5
            and opened.min() >= np.median(held) + 10
            and np.isclose(np.median(held), item["held_aperture_mm"], rtol=0, atol=1e-8)
            and np.isclose(
                opened.min(), item["minimum_open_aperture_mm"], rtol=0, atol=1e-8
            )
        ):
            raise ValueError("measured release evidence does not match source jaws")
