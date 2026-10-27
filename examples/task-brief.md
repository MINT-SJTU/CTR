# New task brief

Copy this file and fill it in before giving it to your coding Agent.

- Language goal and observable completion condition:
- Source video path, SHA-256, FPS, dimensions and camera viewpoint:
- Left/right arm assignment and object identities:
- Observed stages and uncertain/occluded events:
- Dependencies that must hold between arms:
- Permitted overlap, valid waiting states and protected contact intervals:
- Desired output: video only / synchronized LeRobot dataset:
- If dataset: path/revision, split, camera keys, action/state layout and units:
- Robot URDF/meshes, base transforms and camera/depth calibration if available:
- Output/work directories (separate from immutable inputs):
- Visual and numeric acceptance criteria:

Example intent: "The right arm opens the drawer. The left arm picks up a cube,
places it inside, and withdraws. The right arm then closes the drawer. Overlap
pickup with opening, but never insert before sufficient opening or close before
withdrawal." Event frame numbers must be measured from the supplied video.
