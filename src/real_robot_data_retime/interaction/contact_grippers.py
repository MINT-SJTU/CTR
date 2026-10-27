"""Recover a misplaced fingertip from independent object-contact observations."""

import cv2
import numpy as np

from .gripper_geometry import end_effector
from .origin_events import origin_departure_interval


def recover_contact_grippers(frames, proposals, grippers, object_ids, sam, fps):
    """Resegment the hand, excluding the object and distant arm from its prompt.

    Whole-arm masks still determine arm identity and remain unchanged. Only
    observed local hand masks, centers and aperture estimates are replaced.
    """
    n, h, w = frames.shape[:3]
    robots = np.unpackbits(grippers["robot_masks"], axis=-1, count=w).astype(bool)
    union = robots.any(axis=1)
    result = dict(grippers)
    for key in ("centers", "apertures", "masks"):
        result[key] = grippers[key].copy()
    audit = []
    for object_id in object_ids:
        p = proposals[object_id]
        event, _ = origin_departure_interval(frames, p, union, fps)
        if event is None:
            audit.append(dict(object_id=object_id, status="no_observed_contact"))
            continue
        first, last = event["last_observed_at_origin"], event["first_observed_empty"]
        coverage = robots[first:last, :, p["mask"]].mean(axis=(0, 2))
        side = int(np.argmax(coverage))
        if coverage[side] < 0.3 or coverage[side] - coverage[1 - side] < 0.15:
            audit.append(dict(object_id=object_id, status="ambiguous_contact_arm"))
            continue
        seed = max(0, first - round(fps * 0.1))
        start, stop = max(0, first - round(fps * 0.5)), min(n, last + round(fps * 3))
        cx, cy = p["origin"]
        pad = round(w * 0.12)
        x0, y0 = max(0, int(cx) - pad), max(0, int(cy) - pad)
        x1, y1 = min(w, int(cx) + pad), min(h, int(cy) + pad)
        box = np.zeros((h, w), bool)
        box[y0:y1, x0:x1] = True
        excluded = cv2.dilate(
            p["mask"].astype(np.uint8), np.ones((7, 7), np.uint8)
        ).astype(bool)
        support = robots[seed, side] & box & ~excluded
        distance = cv2.distanceTransform(support.astype(np.uint8), cv2.DIST_L2, 5)
        positives = []
        for _ in range(3):
            y, x = np.unravel_index(np.argmax(distance), distance.shape)
            if distance[y, x] < 2:
                break
            positives.append([int(x), int(y)])
            cv2.circle(distance, (int(x), int(y)), 12, 0, -1)
        if not positives:
            audit.append(dict(object_id=object_id, status="no_independent_hand_prompt"))
            continue
        yy, xx = np.where(robots[seed, side] & ~box)
        ids = np.linspace(0, len(xx) - 1, min(12, len(xx)), dtype=int)
        negatives = [[int(xx[i]), int(yy[i])] for i in ids]
        negatives.extend(q["origin"].tolist() for q in proposals)
        prompt = dict(
            bbox=[x0, y0, x1 - x0, y1 - y0],
            mask=support,
            positive_points=positives,
            negative_points=negatives,
        )
        used = []
        for reverse, end in [(False, stop), (True, start - 1)]:
            for t, masks in sam.propagate(
                frames, [prompt], seed_frame=seed, reverse=reverse, stop_frame=end
            ):
                mask = masks[0]
                area = int(mask.sum())
                if (
                    area < 80
                    or area > w * h * 0.15
                    or (mask & robots[t, side]).sum() / area < 0.65
                ):
                    continue
                estimate = end_effector(mask, side)
                if estimate is None:
                    continue
                center, aperture, local = estimate
                result["centers"][t, side] = center
                result["apertures"][t, side] = aperture
                result["masks"][t, side] = local
                used.append(int(t))
        audit.append(
            dict(
                object_id=object_id,
                status="measured" if used else "no_supported_hand_masks",
                arm=["left", "right"][side],
                contact_overlap=coverage.tolist(),
                seed_frame=seed,
                source_window=[start, stop],
                bbox=prompt["bbox"],
                positive_points=positives,
                negative_points=negatives,
                measured_frames=sorted(set(used)),
            )
        )
    return result, audit
