"""Visible-frame per-arm reseeding; context survives empty return boundaries."""

import numpy as np, cv2, gc
from .water_io import dump
from real_robot_data_retime.interaction.video import components
from real_robot_data_retime.interaction.photometric_motion import photometric_motion
from real_robot_data_retime.interaction.neural_tracks import segment_robots
from real_robot_data_retime.interaction.robot_discovery import (
    prompt_from_robot_region,
    robot_mask_audit,
)
from real_robot_data_retime.segmentation.sam_backend import SamVideo


def inside_point(mask, y):
    xs = np.flatnonzero(mask[y])
    assert len(xs)
    return [float(xs[np.argmin(abs(xs - np.median(xs)))]), float(y)]


def prompt(mask, other):
    if not mask.any():
        raise ValueError("No visible pixels for reseeding")
    p = prompt_from_robot_region(mask)
    ys = np.flatnonzero(mask.any(axis=1))
    p["positive_points"] = [
        inside_point(mask, int(ys[round(q * (len(ys) - 1))]))
        for q in [0.05, 0.25, 0.5, 0.75, 0.95]
    ]
    if other.any():
        d = cv2.distanceTransform(other.astype("uint8"), cv2.DIST_L2, 5)
        y, x = np.unravel_index(d.argmax(), d.shape)
        p["negative_points"] = [[float(x), float(y)]]
    return p


def mutually_exclusive(candidate, reference, previous=None):
    """Symmetric nearest-supported ownership. Unresolved ties remain absent."""
    out = candidate.copy() | reference
    for side in [0, 1]:
        out[side] &= ~(reference[1 - side] & ~reference[side])
    overlap = out[0] & out[1]
    if not overlap.any():
        return out, 0
    costs = []
    for side in [0, 1]:
        if not reference[side].any():
            d = np.full(overlap.shape, 1e4, np.float32)
        else:
            d = cv2.distanceTransform(
                (~reference[side]).astype("uint8"), cv2.DIST_L2, 5
            )
        if previous is not None and previous[side].any():
            d += 0.35 * cv2.distanceTransform(
                (~previous[side]).astype("uint8"), cv2.DIST_L2, 5
            )
        costs.append(d)
    out[0][overlap] = costs[0][overlap] + 0.5 < costs[1][overlap]
    out[1][overlap] = costs[1][overlap] + 0.5 < costs[0][overlap]
    return out, int(overlap.sum())


def trim_static_rail(mask, evidence):
    out = mask.copy()
    near = cv2.dilate(evidence.astype("uint8"), np.ones((9, 9), np.uint8)).astype(bool)
    unsupported = mask & ~near
    broad = cv2.morphologyEx(
        unsupported.astype("uint8"), cv2.MORPH_OPEN, np.ones((31, 1), np.uint8)
    )
    thin = unsupported & ~cv2.dilate(broad, np.ones((5, 5), np.uint8)).astype(bool)
    for reg, stat, _ in components(thin, 1):
        # A wide, thin unchanging appendage can connect a visible hand to the
        # static table rail. Keep broad wrist regions even on dark backgrounds.
        if stat[2] > 50 and stat[3] < 30 and stat[2] > 4 * stat[3]:
            out[reg] = False
    return out


def exclude_observed_rail(robots, frames, background, upper_limit):
    """Remove only a wide stationary background band matching an observed plate."""
    h, w = background.shape[:2]
    dark = (cv2.cvtColor(background, cv2.COLOR_BGR2GRAY) < 115).astype("uint8")
    dark[int(upper_limit):] = 0
    from .interaction.evidence import stable_runs
    rows = dark.sum(axis=1) >= w * 0.95
    candidates = [(a, b) for a, b in stable_runs(rows, 1) if b - a < h * 0.15]
    band = np.zeros(h, bool)
    if candidates:
        a, b = max(candidates, key=lambda interval: interval[1])
        pad = max(3, round(h * 0.025))
        band[max(0, a - pad):min(int(upper_limit), b + pad)] = True
    rail = dark.astype(bool) & band[:, None]
    # A one-pixel camera/codec edge displacement must not turn a fixed desk
    # rail into articulated foreground. Every removed color still needs an
    # observed background match in this bounded two-pixel neighborhood.
    jitter = 2
    rail = cv2.dilate(rail.astype("uint8"), np.ones((5, 5), np.uint8)).astype(bool)
    rail[int(upper_limit):] = False
    removed = 0
    reference = np.pad(background.astype(np.int16), ((jitter,jitter),(jitter,jitter),(0,0)), mode="edge")
    for i, frame in enumerate(frames):
        unchanged = np.zeros((h,w),bool)
        observed = frame.astype(np.int16)
        for dy in range(2*jitter+1):
            for dx in range(2*jitter+1):
                unchanged |= np.max(np.abs(observed-reference[dy:dy+h,dx:dx+w]),axis=2)<=18
        excluded = rail & unchanged
        removed += int(robots[i, :, excluded].sum())
        robots[i, :, excluded] = False
    return dict(removed_pixels=removed, observed_rail_pixels=int(rail.sum()), appearance_threshold=18, camera_pixel_uncertainty=2)


def add_entry_fragments(robots, frame):
    """Boundary evidence fills unowned pixels; existing arm tracks keep ownership."""
    if (robots[0] & robots[1]).any():
        raise ValueError("ambiguous arm ownership before boundary supplementation")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    for side in range(2):
        roi = np.zeros((h, w), bool)
        roi[int(h * 0.625) :, : int(w * 110 / 640)] = side == 0
        roi[int(h * 0.625) :, int(w * 530 / 640) :] = side == 1
        for region, _, _ in components((gray < 75) & roi, 25):
            touches_edge = region[:, :2].any() if side == 0 else region[:, -2:].any()
            if touches_edge:
                robots[side] |= region & ~robots[1 - side]


def clean_masks(robots, frames, label, *, observed_background=None):
    """Apply the final observed-background repairs once, before audit/export."""
    gray = np.asarray([cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames])
    old = robots.copy()
    ids = np.argsort(robots.sum(axis=(1, 2, 3)))[:7]
    background = (
        gray[0].astype(float) if label == "pre" else np.median(gray[ids], axis=0)
    )
    static_dark = cv2.dilate(
        (background < 115).astype("uint8"), np.ones((9, 9), np.uint8)
    ).astype(bool)
    for k in range(len(frames)):
        evidence = (background - gray[k] > 18) & (gray[k] < 115)
        if label == "pre":
            evidence &= ~static_dark
        for side in range(2):
            robots[k, side] = trim_static_rail(robots[k, side], evidence)
            if label == "pre":
                for region, stat, _ in components(robots[k, side], 1):
                    if (
                        stat[3] < 40
                        and stat[2] > 3 * stat[3]
                        and (region & evidence).sum() < max(12, stat[4] * 0.12)
                    ):
                        robots[k, side, region] = False
                h, w = gray[k].shape
                roi = np.zeros((h, w), bool)
                roi[int(h * 0.625) :, : int(w * 110 / 640)] = side == 0
                roi[int(h * 0.625) :, int(w * 530 / 640) :] = side == 1
                robots[k, side] |= old[k, side] & roi & (gray[k] < 75)
    if label == "post":
        background_rgb = np.median(frames[ids], axis=0).astype("uint8")
        hsv = cv2.cvtColor(background_rgb, cv2.COLOR_BGR2HSV)
        wood = (
            (hsv[:, :, 0] >= 5)
            & (hsv[:, :, 0] <= 40)
            & (hsv[:, :, 1] > 45)
            & (hsv[:, :, 2] > 35)
        )
        choices = components(wood, 5000)
        if not choices:
            raise ValueError("stationary tray not visible for return-mask cleanup")
        region = max(choices, key=lambda item: item[1][4])[0].astype("uint8")
        contours, _ = cv2.findContours(
            region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        tray = np.zeros_like(region)
        cv2.drawContours(tray, contours, -1, 1, -1)
        tray = cv2.dilate(tray, np.ones((5, 5), np.uint8)).astype(bool)
        for k, frame in enumerate(frames):
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            dark = ((hsv[:, :, 2] < 85) & (hsv[:, :, 1] < 110)) | (gray[k] < 35)
            near = cv2.dilate(dark.astype("uint8"), np.ones((5, 5), np.uint8)).astype(
                bool
            )
            robots[k, :, tray & ~near] = False
            for side in range(2):
                for region, _, _ in components(robots[k, side], 1):
                    if np.all(tray[region]):
                        robots[k, side, region] = False
    rail_audit = None
    if label == "post" and observed_background is not None:
        upper_limit = int(np.flatnonzero(tray.any(axis=1))[0])
        rail_audit = exclude_observed_rail(robots, frames, observed_background, upper_limit)
    return dict(
        removed_pixels=int((old & ~robots).sum()),
        method="observed stationary rail and wooden tray exclusion",
        observed_background_rail=rail_audit,
    )


def segment(O, f, phase, *, phases=("pre", "post")):
    if not set(phases) <= {"pre", "post"}:
        raise ValueError("unknown segmentation phase")
    sam = None
    S, E = phase["shared_start"], phase["shared_end"]
    h, w = f.shape[1:3]
    for label, start, end in [("pre", 0, S + 1), ("post", E, len(f))]:
        if label not in phases:
            continue
        dest = O / (label + "-segmentation.npz")
        print("segment", label, flush=True)
        if sam is None:
            sam = SamVideo("facebook/sam2.1-hiera-large")
        context_start = start if label == "pre" else max(S, E - 80)
        context = f[context_start:end]
        crop = start - context_start
        photo = photometric_motion(context)
        try:
            packed, seeds = segment_robots(context, photo.discovery, sam)
        except ValueError as exc:
            if (
                label != "post"
                or context_start == 0
                or "no entry-anchored articulated foreground" not in str(exc)
            ):
                raise
            # Some tray-return views obscure entry motion even with 80 frames of context.
            # Recover identity from the earlier observed pickup, never invent a seed.
            context_start = 0
            context = f[:end]
            crop = start
            photo = photometric_motion(context)
            packed, seeds = segment_robots(context, photo.discovery, sam)
        reference = np.unpackbits(packed, axis=-1, count=w).astype(bool)
        gray = np.array([cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) for im in context])
        ids = np.argsort(reference.sum(axis=(1, 2, 3)))[:7]
        bg = np.median(gray[ids], axis=0)
        if label == "post":
            # A large tray latch is not a visible-arm seed. Exclude static
            # wood before choosing each arm's largest observed return mask.
            clean_masks(reference[crop:], context[crop:], "post")
        candidate = reference.copy()
        repairs = []
        for side in [0, 1]:
            # Each arm may be invisible at the cut; choose its own visible frame.
            area = reference[:, side].sum(axis=(1, 2))
            seed = (
                crop + int(area[crop:].argmax())
                if label == "post"
                else int(area.argmax())
            )
            if area[seed] < 40:
                raise ValueError(
                    f"No visible arm {side} even with preceding synchronized context"
                )
            full = reference[seed, side].copy()
            near = cv2.dilate(full.astype("uint8"), np.ones((61, 61), np.uint8)).astype(
                bool
            )
            evidence = (bg - gray[seed] > 25) & (gray[seed] < 115)
            for region, _, _ in components(evidence, 50):
                if (region & near).sum() > 50 and not (
                    region & reference[seed, 1 - side]
                ).any():
                    full |= region
            proposal = prompt(full, reference[seed, 1 - side])
            repairs.append(
                dict(
                    side=side,
                    seed=seed + context_start,
                    bbox=proposal["bbox"],
                    positive_points=proposal["positive_points"],
                    method="visible_frame_context_reseed",
                )
            )
            for reverse in [False, True]:
                for k, m in sam.propagate(
                    context, [proposal], seed_frame=seed, reverse=reverse
                ):
                    candidate[k, side] = m[0]
        robots = np.zeros_like(reference)
        conflicts = []
        prev = None
        for k in range(len(context)):
            # Prior supports are from the same frame. Temporal support is advected with
            # observed optical flow before contributing to ambiguous-pixel ownership.
            warped = None
            if prev is not None and (candidate[k, 0] & candidate[k, 1]).any():
                flow = cv2.calcOpticalFlowFarneback(
                    gray[k], gray[k - 1], None, 0.5, 3, 21, 3, 5, 1.2, 0
                )
                yy, xx = np.indices((h, w), dtype="float32")
                warped = np.array(
                    [
                        cv2.remap(
                            x.astype("uint8"),
                            xx + flow[:, :, 0],
                            yy + flow[:, :, 1],
                            cv2.INTER_NEAREST,
                        ).astype(bool)
                        for x in prev
                    ]
                )
            robots[k], count = mutually_exclusive(candidate[k], reference[k], warped)
            if count:
                conflicts.append(dict(source_frame=k + context_start, pixels=count))
            prev = robots[k]
        robots = robots[crop:]
        frames = context[crop:]
        visibility_rejections = []
        if label == "post":
            # A tracker can latch onto the static dark table rail after the arm exits.
            # Require independently observed darkening for each retained component.
            for k in range(len(frames)):
                evidence = (bg - gray[k + crop] > 18) & (gray[k + crop] < 115)
                for side in [0, 1]:
                    robots[k, side] = trim_static_rail(robots[k, side], evidence)
                    for reg, stat, _ in components(robots[k, side], 1):
                        if (reg & evidence).sum() < max(12, stat[4] * 0.12):
                            robots[k, side, reg] = False
                            visibility_rejections.append(
                                dict(
                                    source_frame=k + start,
                                    side=side,
                                    pixels=int(stat[4]),
                                )
                            )
        # Supplement visible boundary pixels without overwriting tracked identity.
        for k, im in enumerate(frames):
            add_entry_fragments(robots[k], im)
        if (robots[:, 0] & robots[:, 1]).any():
            raise ValueError("ambiguous arm ownership")
        objects = np.zeros((0, len(frames), h, w), bool)
        if label == "pre":
            ps = bottle_prompts(frames[0])
            objects = np.zeros((2, len(frames), h, w), bool)
            for k, m in sam.propagate(frames, ps):
                for side in [0, 1]:
                    contours, _ = cv2.findContours(
                        m[side].astype("uint8"),
                        cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE,
                    )
                    filled = np.zeros((h, w), np.uint8)
                    cv2.drawContours(filled, contours, -1, 1, cv2.FILLED)
                    objects[side, k] = filled > 0
        cleanup = clean_masks(
            robots, frames, label,
            observed_background=f[0] if label == "post" and phase["head_removed"] > 0 else None,
        )
        localphoto = photo if crop == 0 else photometric_motion(frames)
        audit = robot_mask_audit(
            np.packbits(robots, axis=-1),
            localphoto.discovery,
            phase["fps"],
            pixel_support=localphoto.support,
        )
        audit.update(
            cleanup=cleanup,
            seeds=seeds,
            repairs=repairs,
            context_start=context_start,
            output_start=start,
            conflicts_resolved=conflicts,
            visibility_rejections=visibility_rejections,
            conflict_policy="same-frame supported identity plus flow-advected temporal distance; no side priority",
            visual_review="pending",
        )
        np.savez_compressed(
            dest,
            robots=np.packbits(robots, axis=-1),
            objects=np.packbits(objects, axis=-1),
            frame_shape=[h, w],
            source_start=start,
            source_stop=end,
        )
        dump(O / (label + "-audit.json"), audit)
        ims = []
        for k in np.linspace(0, len(frames) - 1, 12).astype(int):
            im = frames[k].copy()
            for side, color in enumerate([(0, 255, 255), (255, 120, 0)]):
                im[robots[k, side]] = (
                    im[robots[k, side]] * 0.5 + np.array(color) * 0.5
                ).astype("uint8")
            cv2.putText(im, f"src{k+start}", (8, 26), 0, 0.7, (0, 0, 255), 2)
            ims.append(cv2.resize(im, (320, 240)))
        cv2.imwrite(
            str(O / (label + "-masks-repaired.jpg")),
            np.vstack([np.hstack(ims[i : i + 3]) for i in range(0, 12, 3)]),
        )
        del reference, candidate, robots, objects, photo, localphoto, gray
        gc.collect()
    del sam
    gc.collect()


def bottle_prompts(frame):
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    red = (
        ((hsv[:, :, 0] < 12) | (hsv[:, :, 0] > 170))
        & (hsv[:, :, 1] > 100)
        & (hsv[:, :, 2] > 80)
    )
    red[: int(h * 0.25)] = False
    red[int(h * 0.68) :] = False
    prompts = []
    for side in [0, 1]:
        candidates = [
            x
            for x in components(red, 50)
            if (x[2][0] < w * 0.35 if side == 0 else x[2][0] > w * 0.65)
        ]
        if not candidates:
            raise ValueError("No unambiguous side bottle cap")
        cap, st, (cx, cy) = min(candidates, key=lambda x: x[2][1])
        scale = max(st[2], st[3])
        x0 = max(0, int(cx - 2.8 * scale))
        x1 = min(w, int(cx + 2.8 * scale))
        y0 = max(0, int(cy - scale))
        y1 = min(h, int(cy + 4.5 * scale))
        roi = np.zeros((h, w), bool)
        roi[int(cy + 10) : y1, x0:x1] = True
        teal = roi & (hsv[:, :, 1] > 65) & (hsv[:, :, 2] > 40)
        labels = components(teal, 40)
        if not labels:
            raise ValueError("Bottle full-body label prompt missing")
        label, _, (lx, ly) = max(labels, key=lambda x: x[1][4])
        points = [[float(cx), float(cy)], [float(lx), float(ly)]]
        yy, xx = np.where(label)
        for q in [0.2, 0.8]:
            y = int(np.quantile(yy, q))
            xs = xx[abs(yy - y) <= 2]
            points.append([float(np.median(xs)), float(y)])
        prompts.append(
            dict(
                mask=cap | label,
                bbox=[x0, y0, x1 - x0, y1 - y0],
                positive_points=points,
                negative_points=[],
            )
        )
    return prompts
