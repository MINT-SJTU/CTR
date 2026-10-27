from __future__ import annotations

import gzip
from pathlib import Path
from typing import Any, Protocol

from flask import Flask, Response, jsonify, request, send_file


class Backend(Protocol):
    """Interface implemented by concrete robot/dataset viewers."""

    def config(self) -> dict[str, Any]:
        ...

    def episode_payload(
        self,
        dataset: str,
        episode_index: int,
        signal: str | None = None,
    ) -> dict[str, Any]:
        ...

    def transforms(
        self,
        dataset: str,
        episode_index: int,
        signal: str | None = None,
    ) -> bytes:
        ...

    def mesh_path(self, mesh_index: int) -> Path:
        ...

    def video_path(self, dataset: str, episode_index: int, camera: str) -> Path:
        ...

    def mesh_asset_path(self, filename: str) -> Path:
        ...


MESH_MIMETYPES = {
    ".dae": "model/vnd.collada+xml",
    ".stl": "model/stl",
}


def create_app(
    backend: Backend,
    static_dir: str | Path,
    *,
    title: str = "robo-visualize",
) -> Flask:
    static = Path(static_dir).resolve()
    app = Flask(__name__, static_folder=str(static), static_url_path="/static")
    app.json.sort_keys = False
    app.config["viewer_backend"] = backend

    @app.get("/")
    def index() -> Response:
        return send_file(static / "index.html")

    @app.get("/api/config")
    def config() -> Response:
        return jsonify({**backend.config(), "title": title})

    @app.get("/api/episode/<dataset>/<int:episode_index>")
    def episode(dataset: str, episode_index: int) -> Response:
        signal = request.args.get("signal")
        return jsonify(backend.episode_payload(dataset, episode_index, signal))

    @app.get("/api/transforms/<dataset>/<int:episode_index>.bin")
    def transforms(dataset: str, episode_index: int) -> Response:
        signal = request.args.get("signal")
        payload = backend.transforms(dataset, episode_index, signal)
        response = Response(payload, mimetype="application/octet-stream")
        response.headers["Cache-Control"] = "private, max-age=3600"
        return response

    @app.get("/api/mesh/<filename>")
    def mesh(filename: str) -> Response:
        if Path(filename).suffix:
            try:
                index = int(Path(filename).stem)
            except ValueError:
                index = None
            if index is not None:
                path = backend.mesh_path(index)
                response = Response(
                    gzip.compress(path.read_bytes(), compresslevel=5),
                    mimetype=MESH_MIMETYPES.get(path.suffix.lower(), "application/octet-stream"),
                )
                response.headers["Content-Encoding"] = "gzip"
                response.headers["Cache-Control"] = "public, max-age=86400"
                response.headers["Vary"] = "Accept-Encoding"
                return response
        return send_file(backend.mesh_asset_path(filename), conditional=True)

    @app.get("/api/video/<dataset>/<int:episode_index>/<camera>.mp4")
    def video(dataset: str, episode_index: int, camera: str) -> Response:
        path = backend.video_path(dataset, episode_index, camera)
        return send_file(path, mimetype="video/mp4", conditional=True)

    @app.get("/api/ee_trajectory/<dataset>/<int:episode_index>.bin")
    def ee_trajectory(dataset: str, episode_index: int) -> Response:
        if not hasattr(backend, "ee_trajectory"):
            return Response("not found", status=404)
        signal = request.args.get("signal")
        payload = backend.ee_trajectory(dataset, episode_index, signal)
        response = Response(payload, mimetype="application/octet-stream")
        response.headers["Cache-Control"] = "private, max-age=3600"
        return response

    @app.get("/api/perturbation_points.json")
    def perturbation_points() -> Response:
        if not hasattr(backend, "perturbation_points"):
            return Response("not found", status=404)
        return jsonify(backend.perturbation_points())

    @app.get("/api/recovery_vectors.json")
    def recovery_vectors() -> Response:
        if not hasattr(backend, "recovery_vectors"):
            return Response("not found", status=404)
        return jsonify(backend.recovery_vectors())

    return app
