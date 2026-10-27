# Robot geometry provenance

`piper_x_description.urdf` is the CTR snapshot retained for the drawer exporter.
Its source and SHA-256 are recorded in `provenance.json`. Relative to the original
RoboVisualize description it uses the supplied gripper mount rotation
`rpy="0 0 1.5708"`, normalized LF line endings and stripped trailing whitespace.

The original RoboVisualize PiperX description and this CTR description are kept
as distinct calibrated inputs. Mesh files are resolved against the explicit
`--mesh-root` from the verified asset downloader. The collision checker defaults
to 0.59 m inter-arm base spacing; verify the actual geometry for each recording.

Public robot asset repositories, fixed revisions, exact file hashes and modified
snapshots are listed in `robot-sources.json`. See
[robot asset setup](../docs/robot-assets.md) and [third-party notices](../THIRD_PARTY.md).
