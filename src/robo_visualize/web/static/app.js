import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { ColladaLoader } from "three/addons/loaders/ColladaLoader.js";
import { STLLoader } from "three/addons/loaders/STLLoader.js";

const DEFAULT_COLORS = [0x4d9fe8, 0xed883f];
const PHASE_COLORS = {
  async: [0x38e891, 0x38e891],
  sync: [0x9b87f5, 0x9b87f5],
};

const elements = {
  scene: document.getElementById("scene"),
  dataset: document.getElementById("dataset-select"),
  episode: document.getElementById("episode-select"),
  signalControl: document.getElementById("signal-control"),
  signal: document.getElementById("signal-select"),
  refreshDatasets: document.getElementById("refresh-datasets"),
  resetView: document.getElementById("reset-view"),
  play: document.getElementById("play-button"),
  timeline: document.getElementById("timeline"),
  rate: document.getElementById("rate-select"),
  time: document.getElementById("time-label"),
  frame: document.getElementById("frame-label"),
  phase: document.getElementById("phase-label"),
  meta: document.getElementById("meta-label"),
  trajectoryLabel: document.getElementById("trajectory-label"),
  status: document.getElementById("status"),
  armLegend: document.getElementById("arm-legend"),
  cameraGrid: document.getElementById("camera-grid"),
  title: document.getElementById("viewer-title"),
  eyebrow: document.getElementById("viewer-eyebrow"),
};

const state = {
  config: null,
  episode: null,
  transforms: null,
  eeTrajectoryLine: null,
  perturbationPoints: null,
  sidRecoveryGroup: null,
  sidRecoveryMetrics: null,
  sidRecoveryCount: 0,
  liveRecoveryGroup: null,
  livePollTimer: null,
  livePollBusy: false,
  meshes: [],
  videos: new Map(),
  videoInfo: new Map(),
  masterCamera: null,
  materials: [],
  currentFrame: -1,
  loadGeneration: 0,
  scrubbing: false,
  signal: null,
};

const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.shadowMap.enabled = true;
elements.scene.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.fog = new THREE.Fog(0x09131c, 1.0, 3.5);

const camera = new THREE.PerspectiveCamera(43, 1, 0.01, 20);
camera.up.set(0, 0, 1);
camera.position.set(1.1, -1.3, 1.1);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.minDistance = 0.25;
controls.maxDistance = 6.0;
controls.target.set(0, -0.2, 0.25);

scene.add(new THREE.HemisphereLight(0xd8efff, 0x17212a, 2.3));
const keyLight = new THREE.DirectionalLight(0xffffff, 3.0);
keyLight.position.set(-0.6, -0.7, 1.1);
keyLight.castShadow = true;
scene.add(keyLight);

const floor = new THREE.Mesh(
  new THREE.PlaneGeometry(1.5, 1.2),
  new THREE.MeshStandardMaterial({ color: 0x1a2934, roughness: 0.94, metalness: 0.02 }),
);
floor.position.z = -0.012;
floor.receiveShadow = true;
scene.add(floor);

const grid = new THREE.GridHelper(1.4, 24, 0x375166, 0x1a3040);
grid.rotation.x = Math.PI / 2;
grid.position.z = -0.008;
grid.material.opacity = 0.4;
grid.material.transparent = true;
scene.add(grid);

const robotGroup = new THREE.Group();
scene.add(robotGroup);
const overlayGroup = new THREE.Group();
scene.add(overlayGroup);

function resetCamera() {
  camera.position.set(1.1, -1.3, 1.1);
  controls.target.set(0, -0.2, 0.25);
  controls.update();
}

resetCamera();

const resizeObserver = new ResizeObserver(() => {
  const width = elements.scene.clientWidth;
  const height = elements.scene.clientHeight;
  if (!width || !height) return;
  renderer.setSize(width, height, false);
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
});
resizeObserver.observe(elements.scene);

function fetchChecked(url, responseType = "json") {
  return fetch(url).then((response) => {
    if (!response.ok) throw new Error(response.status + " " + response.statusText + ": " + url);
    return responseType === "arrayBuffer" ? response.arrayBuffer() : response.json();
  });
}

function configState() {
  return {
    loader: state.config.loader || (state.config.meshes[0]?.url.endsWith(".dae") ? "collada" : "stl"),
    sides: state.config.sides || 1,
    sideLabels: state.config.sideLabels || state.config.sides || ["arm"],
    meshCount: state.config.mesh_count,
    colors: state.config.colors || DEFAULT_COLORS,
  };
}

function materialForLink(linkName, materials) {
  if (!linkName) return materials[0];
  for (let index = 0; index < state.config.sideLabels.length; index += 1) {
    const prefix = state.config.sideLabels[index] + "_";
    if (linkName.startsWith(prefix)) return materials[index];
  }
  return materials[materials.length - 1] || materials[0];
}

function colladaMeshInfo(meshInfo, cfg) {
  const suffix = meshInfo.url.toLowerCase().split("?")[0];
  if (suffix.endsWith(".dae")) return true;
  if (suffix.endsWith(".stl")) return false;
  return cfg.loader === "collada";
}

async function loadRobotMeshes() {
  const cfg = configState();
  const colladaLoader = new ColladaLoader();
  const stlLoader = new STLLoader();
  const uniquePromises = new Map();
  for (const meshInfo of state.config.meshes) {
    if (!uniquePromises.has(meshInfo.url)) {
      const isCollada = colladaMeshInfo(meshInfo, cfg);
      const loader = isCollada ? colladaLoader : stlLoader;
      uniquePromises.set(
        meshInfo.url,
        loader.loadAsync(meshInfo.url).then((loaded) => {
          if (isCollada) return loaded.scene;
          loaded.computeVertexNormals();
          return loaded;
        }),
      );
    }
  }

  const sideCount = Array.isArray(cfg.sides) ? cfg.sides.length : 1;
  const materials = Array.from({ length: Math.max(sideCount, 1) }, (_, index) =>
    new THREE.MeshStandardMaterial({
      color: cfg.colors[index % cfg.colors.length],
      roughness: 0.72,
      metalness: 0.05,
    }),
  );
  state.materials = materials;

  if (sideCount > 1) state.meshes = [[], []];
  for (const meshInfo of state.config.meshes) {
    const loaded = await uniquePromises.get(meshInfo.url);
    const isCollada = colladaMeshInfo(meshInfo, cfg);
    for (let side = 0; side < Math.max(sideCount, 1); side += 1) {
      const content = loaded.clone(true);
      if (isCollada) {
        content.traverse((child) => {
          if (!child.isMesh) return;
          child.material = materialForLink(meshInfo.link_name, materials);
          child.castShadow = true;
          child.receiveShadow = true;
        });
        content.rotation.set(0, 0, 0);
        content.scale.set(...meshInfo.scale);
        content.updateMatrix();
        const wrapper = new THREE.Group();
        wrapper.add(content);
        wrapper.matrixAutoUpdate = false;
        if (sideCount > 1) state.meshes[side].push(wrapper);
        else state.meshes.push(wrapper);
        robotGroup.add(wrapper);
      } else {
        const geometry = loaded.clone();
        geometry.scale(...meshInfo.scale);
        const mesh = new THREE.Mesh(geometry, materials[side]);
        mesh.matrixAutoUpdate = false;
        mesh.castShadow = true;
        mesh.receiveShadow = true;
        if (sideCount > 1) state.meshes[side].push(mesh);
        else state.meshes.push(mesh);
        robotGroup.add(mesh);
      }
    }
  }

  elements.armLegend.replaceChildren();
  const labels = Array.isArray(cfg.sideLabels) ? cfg.sideLabels : ["arm"];
  labels.forEach((label, index) => {
    const span = document.createElement("span");
    const swatch = document.createElement("i");
    swatch.style.background = "#" + cfg.colors[index % cfg.colors.length].toString(16).padStart(6, "0");
    span.append(swatch, document.createTextNode(String(label)));
    elements.armLegend.appendChild(span);
  });
}

function clearOverlayTrajectory() {
  if (state.eeTrajectoryLine) {
    overlayGroup.remove(state.eeTrajectoryLine);
    state.eeTrajectoryLine.geometry.dispose();
    state.eeTrajectoryLine.material.dispose();
    state.eeTrajectoryLine = null;
  }
}

function setupPerturbationOverlay(config) {
  if (state.perturbationPoints) {
    overlayGroup.remove(state.perturbationPoints);
    state.perturbationPoints.geometry.dispose();
    state.perturbationPoints.material.dispose();
    state.perturbationPoints = null;
  }
  const overlay = config.overlay;
  if (!overlay || !overlay.has_perturbations || !overlay.points_url) return;
  fetchChecked(overlay.points_url).then((payload) => {
    const rows = payload.points || [];
    if (!rows.length) return;
    const positions = new Float32Array(rows.length * 3);
    rows.forEach((point, index) => {
      positions[index * 3] = point.position[0];
      positions[index * 3 + 1] = point.position[1];
      positions[index * 3 + 2] = point.position[2];
    });
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    const material = new THREE.PointsMaterial({
      color: 0xff8a3d,
      size: 0.012,
      sizeAttenuation: true,
      transparent: true,
      opacity: 0.9,
    });
    state.perturbationPoints = new THREE.Points(geometry, material);
    overlayGroup.add(state.perturbationPoints);
  }).catch(showError);
}

function disposeObject(root) {
  root.traverse((child) => {
    child.geometry?.dispose();
    const materials = Array.isArray(child.material) ? child.material : [child.material];
    materials.filter(Boolean).forEach((material) => {
      material.map?.dispose();
      material.dispose();
    });
  });
}

function trajectoryLine(poses, color, opacity) {
  const positions = new Float32Array(poses.length * 3);
  poses.forEach((pose, index) => {
    positions[index * 3] = pose[0];
    positions[index * 3 + 1] = pose[1];
    positions[index * 3 + 2] = pose[2];
  });
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  return new THREE.Line(
    geometry,
    new THREE.LineBasicMaterial({ color, transparent: opacity < 1, opacity }),
  );
}

function composePoses(parentPose, childPose) {
  const parentPosition = new THREE.Vector3(...parentPose.slice(0, 3));
  const parentOrientation = new THREE.Quaternion(
    parentPose[4], parentPose[5], parentPose[6], parentPose[3],
  );
  const childPosition = new THREE.Vector3(...childPose.slice(0, 3));
  childPosition.applyQuaternion(parentOrientation).add(parentPosition);
  const childOrientation = new THREE.Quaternion(
    childPose[4], childPose[5], childPose[6], childPose[3],
  );
  parentOrientation.multiply(childOrientation).normalize();
  return [
    childPosition.x,
    childPosition.y,
    childPosition.z,
    parentOrientation.w,
    parentOrientation.x,
    parentOrientation.y,
    parentOrientation.z,
  ];
}

function recoveryPoint(pose, color, radius) {
  const point = new THREE.Mesh(
    new THREE.SphereGeometry(radius, 20, 12),
    new THREE.MeshStandardMaterial({ color, roughness: 0.42, metalness: 0.05 }),
  );
  point.position.set(pose[0], pose[1], pose[2]);
  return point;
}

function recoveryLabel(sampleIndex, pose) {
  const canvas = document.createElement("canvas");
  canvas.width = 96;
  canvas.height = 48;
  const context = canvas.getContext("2d");
  if (!context) throw new Error("2D canvas context is unavailable");
  context.fillStyle = "rgba(8, 19, 29, 0.88)";
  context.fillRect(1, 1, 94, 46);
  context.strokeStyle = "#ff7f32";
  context.lineWidth = 3;
  context.strokeRect(1.5, 1.5, 93, 45);
  context.fillStyle = "#f3f8fb";
  context.font = "600 25px system-ui, sans-serif";
  context.textAlign = "center";
  context.textBaseline = "middle";
  context.fillText("P" + (sampleIndex + 1), 48, 25);
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  const label = new THREE.Sprite(
    new THREE.SpriteMaterial({ map: texture, transparent: true, depthTest: false }),
  );
  label.position.set(pose[0], pose[1], pose[2] + 0.018);
  label.scale.set(0.038, 0.019, 1);
  label.renderOrder = 20;
  return label;
}

function setObjectOpacity(root, opacity) {
  root.traverse((child) => {
    const materials = Array.isArray(child.material) ? child.material : [child.material];
    materials.filter(Boolean).forEach((material) => {
      material.transparent = true;
      material.opacity = opacity;
    });
  });
}

function recoveryArrow(startPose, endPose, color) {
  const start = new THREE.Vector3(startPose[0], startPose[1], startPose[2]);
  const end = new THREE.Vector3(endPose[0], endPose[1], endPose[2]);
  const direction = end.clone().sub(start);
  const length = direction.length();
  if (length < 1e-8) return null;
  direction.normalize();
  const headLength = Math.min(0.025, Math.max(0.008, length * 0.2));
  const headWidth = Math.min(0.014, Math.max(0.004, length * 0.1));
  return new THREE.ArrowHelper(direction, start, length, color, headLength, headWidth);
}

function thickRecoveryArrow(startPose, endPose, color) {
  const start = new THREE.Vector3(startPose[0], startPose[1], startPose[2]);
  const end = new THREE.Vector3(endPose[0], endPose[1], endPose[2]);
  const direction = end.clone().sub(start);
  const length = direction.length();
  if (length < 1e-8) return null;
  direction.normalize();
  const headLength = Math.min(0.012, Math.max(0.002, length * 0.32));
  const shaftLength = Math.max(length - headLength, length * 0.25);
  const radius = Math.min(0.003, Math.max(0.0012, length * 0.07));
  const orientation = new THREE.Quaternion().setFromUnitVectors(
    new THREE.Vector3(0, 1, 0),
    direction,
  );
  const material = new THREE.MeshStandardMaterial({
    color,
    emissive: color,
    emissiveIntensity: 0.22,
    roughness: 0.38,
  });
  const group = new THREE.Group();
  const shaft = new THREE.Mesh(
    new THREE.CylinderGeometry(radius, radius, shaftLength, 12),
    material,
  );
  shaft.quaternion.copy(orientation);
  shaft.position.copy(start).addScaledVector(direction, shaftLength / 2);
  const head = new THREE.Mesh(
    new THREE.ConeGeometry(radius * 2.5, headLength, 16),
    material.clone(),
  );
  head.quaternion.copy(orientation);
  head.position.copy(end).addScaledVector(direction, -headLength / 2);
  group.add(shaft, head);
  return group;
}

function appendOverlayLegend(label, color) {
  const span = document.createElement("span");
  span.dataset.sidOverlay = "true";
  const swatch = document.createElement("i");
  swatch.style.background = "#" + color.toString(16).padStart(6, "0");
  span.append(swatch, document.createTextNode(label));
  elements.armLegend.appendChild(span);
}

function applyLiveTransforms(transforms) {
  const meshCount = state.config.mesh_count;
  if (!Array.isArray(transforms) || transforms.length !== meshCount) {
    throw new Error("live PiperX transform count mismatch");
  }
  transforms.forEach((matrix, meshIndex) => {
    if (!Array.isArray(matrix) || matrix.length !== 16) {
      throw new Error("invalid live PiperX transform matrix");
    }
    const target = state.meshes[meshIndex];
    if (!target) throw new Error("live PiperX mesh is missing at index " + meshIndex);
    target.matrix.fromArray(matrix);
    target.matrixWorldNeedsUpdate = true;
  });
}

function clearLiveRecoveryOverlay() {
  if (!state.liveRecoveryGroup) return;
  overlayGroup.remove(state.liveRecoveryGroup);
  disposeObject(state.liveRecoveryGroup);
  state.liveRecoveryGroup = null;
}

function liveMetric(value) {
  return Number.isFinite(value) ? Number(value).toFixed(3) : "—";
}

function updateLiveState(payload) {
  if (payload.schema !== "up_vla.sid_live_visualization.v2") {
    throw new Error("unexpected live SID visualization schema");
  }
  applyLiveTransforms(payload.mesh_transforms_column_major);
  clearLiveRecoveryOverlay();

  const currentPose = payload.current_tcp_pose_base_wxyz;
  const endpointPose = payload.predicted_endpoint_tcp_pose_base_wxyz;
  const nearestPose = payload.nearest_expert_tcp_pose_base_wxyz;
  const takeoverActive = payload.takeover?.active === true;
  const sourceAgeS = Number(payload.source_age_s);
  const sourceStale = !Number.isFinite(sourceAgeS)
    || sourceAgeS > Number(state.config.live_stale_after_s || 2);
  const recoveryFresh = !takeoverActive
    && !sourceStale
    && (
      payload.takeover == null
      || payload.recovery_takeover_generation === payload.takeover.generation
    );
  if (!Array.isArray(currentPose) || currentPose.length !== 7) {
    throw new Error("live PiperX TCP pose is missing");
  }
  const group = new THREE.Group();
  group.name = "live-sid-recovery";
  const currentPoint = recoveryPoint(currentPose, 0xffffff, 0.0075);
  currentPoint.name = "live-current-ee-tcp";
  group.add(currentPoint);
  if (recoveryFresh && endpointPose) {
    const endpointPoint = recoveryPoint(endpointPose, 0x47f08b, 0.0065);
    endpointPoint.name = "live-predicted-recovery-endpoint";
    group.add(endpointPoint);
    const predictedArrow = thickRecoveryArrow(currentPose, endpointPose, 0x47f08b);
    if (predictedArrow) {
      predictedArrow.name = "live-predicted-recovery-direction";
      group.add(predictedArrow);
    }
  }
  if (recoveryFresh && nearestPose) {
    const nearestPoint = recoveryPoint(nearestPose, 0xffd166, 0.006);
    nearestPoint.name = "live-nearest-expert-point";
    group.add(nearestPoint);
    const referenceArrow = recoveryArrow(currentPose, nearestPose, 0xffd166);
    if (referenceArrow) {
      referenceArrow.name = "live-nearest-expert-reference";
      setObjectOpacity(referenceArrow, 0.45);
      group.add(referenceArrow);
    }
  }
  overlayGroup.add(group);
  state.liveRecoveryGroup = group;

  const ageMs = Math.round(1000 * sourceAgeS);
  const phase = String(payload.phase || "unknown");
  elements.status.textContent = sourceStale
    ? "STALE - policy publisher offline"
    : takeoverActive
    ? "TAKEOVER · live feedback"
    : "LIVE · " + phase + " · " + ageMs + " ms";
  elements.frame.textContent = "attempt " + String(payload.attempt_index).padStart(4, "0");
  elements.phase.hidden = false;
  elements.phase.textContent = takeoverActive ? "MANUAL TAKEOVER" : phase.toUpperCase();
  elements.meta.textContent = "nearest frame "
    + (payload.nearest_expert_frame ?? "—")
    + " · d " + liveMetric(payload.distance_before)
    + " → " + liveMetric(payload.distance_after);
}

async function pollLiveState() {
  if (!state.config?.live_mode || state.livePollBusy) return;
  state.livePollBusy = true;
  try {
    const separator = state.config.live_state_url.includes("?") ? "&" : "?";
    const payload = await fetchChecked(
      state.config.live_state_url + separator + "ts=" + Date.now(),
    );
    updateLiveState(payload);
  } catch (error) {
    console.error(error);
    elements.status.textContent = "Live update failed";
  } finally {
    state.livePollBusy = false;
  }
}

function startLivePolling() {
  document.body.classList.add("live-mode");
  elements.dataset.parentElement.hidden = true;
  elements.episode.parentElement.hidden = true;
  elements.cameraGrid.hidden = true;
  elements.trajectoryLabel.textContent = "Expert trajectory + live SID recovery";
  appendOverlayLegend("expert EE TCP trajectory", 0x5ad0ff);
  appendOverlayLegend("live EE TCP", 0xffffff);
  appendOverlayLegend("policy endpoint / recovery", 0x47f08b);
  appendOverlayLegend("nearest expert", 0xffd166);
  pollLiveState();
  state.livePollTimer = window.setInterval(
    pollLiveState,
    Number(state.config.live_poll_interval_ms || 100),
  );
}

async function setupSIDRecoveryOverlay(config) {
  elements.armLegend.querySelectorAll('[data-sid-overlay="true"]').forEach((item) => item.remove());
  if (state.sidRecoveryGroup) {
    overlayGroup.remove(state.sidRecoveryGroup);
    disposeObject(state.sidRecoveryGroup);
    state.sidRecoveryGroup = null;
    state.sidRecoveryMetrics = null;
    state.sidRecoveryCount = 0;
  }
  const overlay = config.overlay;
  if (!overlay?.has_sid_recovery || !overlay.sid_recovery_url) return;
  const payload = await fetchChecked(overlay.sid_recovery_url);
  const manifold = payload.manifold;
  const samples = payload.samples;
  const full = manifold.full_trajectory_poses_base_link6_wxyz;
  const approach = manifold.approach_trajectory_poses_base_link6_wxyz;
  const controlledToDisplay = overlay.sid_controlled_frame_to_display_pose_wxyz;
  if (!full?.length || !approach?.length) throw new Error("empty SID recovery trajectory");
  if (!samples?.length) throw new Error("empty SID recovery sample set");
  if (overlay.sid_display_frame !== "tcp") {
    throw new Error("SID display frame must be tcp");
  }
  if (!Array.isArray(controlledToDisplay) || controlledToDisplay.length !== 7) {
    throw new Error("SID controlled-frame display transform is missing");
  }
  const displayPose = (pose) => composePoses(pose, controlledToDisplay);

  const group = new THREE.Group();
  group.name = "sid-recovery-overlay";
  const approachLine = trajectoryLine(approach.map(displayPose), 0x31e3d1, 0.98);
  approachLine.name = "sid-approach-manifold";
  group.add(approachLine);

  [...samples]
    .sort((left, right) => left.sample_index - right.sample_index)
    .forEach((sample) => {
      const sampleGroup = new THREE.Group();
      const sampleNumber = sample.sample_index + 1;
      sampleGroup.name = "sid-recovery-sample-" + sampleNumber;
      const pPose = displayPose(sample.p_pose_base_link6_wxyz);
      const qPose = displayPose(sample.attached_episode_q_pose_base_link6_wxyz);
      const predictedEndpoint = displayPose(sample.predicted_endpoint_pose_base_link6_wxyz);
      const pPoint = recoveryPoint(pPose, 0xff7f32, 0.0065);
      pPoint.name = "off-manifold-p-" + sampleNumber;
      const qPoint = recoveryPoint(qPose, 0x31e3d1, 0.005);
      qPoint.name = "nearest-expert-q-" + sampleNumber;
      const endpointPoint = recoveryPoint(predictedEndpoint, 0x47f08b, 0.005);
      endpointPoint.name = "predicted-recovery-endpoint-" + sampleNumber;
      const label = recoveryLabel(sample.sample_index, pPose);
      label.name = "perturbation-label-p-" + sampleNumber;
      sampleGroup.add(pPoint, qPoint, endpointPoint, label);

      const predictedArrow = recoveryArrow(pPose, predictedEndpoint, 0x47f08b);
      const referenceArrow = recoveryArrow(pPose, qPose, 0xffd166);
      if (predictedArrow) {
        predictedArrow.name = "predicted-recovery-direction-" + sampleNumber;
        sampleGroup.add(predictedArrow);
      }
      if (referenceArrow) {
        referenceArrow.name = "nearest-expert-reference-" + sampleNumber;
        setObjectOpacity(referenceArrow, 0.62);
        sampleGroup.add(referenceArrow);
      }
      group.add(sampleGroup);
    });

  overlayGroup.add(group);
  state.sidRecoveryGroup = group;
  state.sidRecoveryMetrics = payload.aggregate_metrics;
  state.sidRecoveryCount = samples.length;
  appendOverlayLegend("actual EE TCP trajectory", 0x5ad0ff);
  appendOverlayLegend("P1–P" + samples.length + " perturbations", 0xff7f32);
  appendOverlayLegend("predicted endpoint / vector", 0x47f08b);
  appendOverlayLegend("nearest expert Q / reference", 0xffd166);
  appendOverlayLegend("SID approach", 0x31e3d1);
}

async function loadEETrajectory(episode, generation) {
  clearOverlayTrajectory();
  if (!episode.ee_trajectory_url) return;
  const buffer = await fetchChecked(signalUrl(episode.ee_trajectory_url), "arrayBuffer");
  if (generation !== state.loadGeneration) return;
  const positions = new Float32Array(buffer);
  if (positions.length % 3 !== 0 || !positions.length) {
    throw new Error("invalid EE trajectory payload");
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  const material = new THREE.LineBasicMaterial({
    color: 0x5ad0ff,
    transparent: true,
    opacity: 0.85,
    linewidth: 2,
  });
  state.eeTrajectoryLine = new THREE.Line(geometry, material);
  overlayGroup.add(state.eeTrajectoryLine);
}

function setupCameraGrid(episode) {
  elements.cameraGrid.replaceChildren();
  state.videos = new Map();
  state.videoInfo = new Map();
  const cameras = state.config.cameras || Object.keys(episode.videos);
  cameras.forEach((name, index) => {
    const card = document.createElement("article");
    card.className = "panel camera-card" + (index === 0 ? " hero" : "");
    const title = document.createElement("div");
    title.className = "panel-title";
    const label = document.createElement("span");
    label.textContent = name;
    const sync = document.createElement("span");
    sync.className = "sync-state";
    sync.textContent = index === 0 ? "MASTER" : "SYNC";
    title.append(label, sync);
    const video = document.createElement("video");
    video.muted = true;
    video.playsInline = true;
    video.preload = "metadata";
    card.append(title, video);
    elements.cameraGrid.appendChild(card);
    video.addEventListener("click", togglePlayback);
    const info = episode.videos[name];
    const fromS = typeof info === "object" && info !== null ? Number(info.from_s || 0) : 0;
    const url = typeof info === "object" && info !== null ? info.url : info;
    video.src = url;
    video.playbackRate = Number(elements.rate.value);
    video.load();
    state.videos.set(name, video);
    state.videoInfo.set(name, { video, fromS, master: index === 0 });
  });
  state.masterCamera = state.config.master_camera || cameras[0];
}

function populateDatasets(preferred = "") {
  const names = Object.keys(state.config.datasets);
  elements.dataset.replaceChildren(
    ...names.map((name) => {
      const option = document.createElement("option");
      const info = state.config.datasets[name];
      option.value = name;
      option.textContent = info.episode_count !== undefined
        ? name + " · " + info.episode_count + " episodes"
        : name;
      return option;
    }),
  );
  if (names.length) elements.dataset.value = names.includes(preferred) ? preferred : names[0];
}

function populateEpisodes(preferred = state.config.preferred_episode_index || 0) {
  const dataset = state.config.datasets[elements.dataset.value];
  let indices = dataset.episode_indices;
  if (!indices) indices = Array.from({ length: dataset.episode_count }, (_, index) => index);
  elements.episode.replaceChildren(
    ...indices.map((episodeIndex) => {
      const option = document.createElement("option");
      option.value = String(episodeIndex);
      option.textContent = "Episode " + String(episodeIndex).padStart(2, "0");
      return option;
    }),
  );
  elements.episode.value = String(indices.includes(preferred) ? preferred : indices[0]);
}

function populateSignal() {
  const signals = state.config.trajectorySignals || state.config.trajectory_signals || [];
  if (signals.length <= 1) {
    elements.signalControl.hidden = true;
    state.signal = null;
    return;
  }
  elements.signalControl.hidden = false;
  elements.signal.replaceChildren(
    ...signals.map((signal) => {
      const option = document.createElement("option");
      option.value = signal;
      option.textContent = signal;
      return option;
    }),
  );
  const preferred = state.config.defaultTrajectorySignal || state.config.default_trajectory_signal;
  elements.signal.value = signals.includes(preferred) ? preferred : signals[0];
  state.signal = elements.signal.value;
}

function waitForMetadata(video) {
  if (video.readyState >= HTMLMediaElement.HAVE_METADATA) return Promise.resolve();
  return new Promise((resolve, reject) => {
    video.addEventListener("loadedmetadata", resolve, { once: true });
    video.addEventListener("error", () => reject(new Error("video metadata failed: " + video.src)), { once: true });
  });
}

function signalUrl(source) {
  let url = source;
  if (state.signal) {
    url += (url.includes("?") ? "&" : "?") + "signal=" + encodeURIComponent(state.signal);
  }
  return url;
}

function transformsUrl(episode) {
  return signalUrl(episode.transforms_url);
}

function formatTime(seconds) {
  const minutes = Math.floor(seconds / 60);
  const remainder = seconds - minutes * 60;
  return String(minutes).padStart(2, "0") + ":" + remainder.toFixed(3).padStart(6, "0");
}

function updateAtTime(seconds) {
  if (!state.episode) return;
  const local = THREE.MathUtils.clamp(seconds, 0, state.episode.duration_s);
  const frame = Math.min(Math.floor(local * state.episode.fps), state.episode.frame_count - 1);
  applyFrame(frame);
  if (!state.scrubbing) elements.timeline.value = String(local);
  elements.time.textContent = formatTime(local) + " / " + formatTime(state.episode.duration_s);
}

function videoLocal(video) {
  const entries = Array.from(state.videoInfo.values()).find((entry) => entry.video === video);
  return entries ? Math.max(0, video.currentTime - entries.fromS) : video.currentTime;
}

function localTime() {
  if (!state.masterCamera) return 0;
    const entry = state.videoInfo.get(state.masterCamera);
  if (!entry) return 0;
  return THREE.MathUtils.clamp(entry.video.currentTime - entry.fromS, 0, state.episode.duration_s);
}

function setVideoLocal(video, local) {
  const entry = Array.from(state.videoInfo.values()).find((item) => item.video === video);
  if (!entry) return;
  const desired = entry.fromS + local;
  if (Math.abs(video.currentTime - desired) > 0.002) video.currentTime = desired;
}

function setLocalTime(seconds) {
  if (!state.episode) return;
  const local = THREE.MathUtils.clamp(seconds, 0, state.episode.duration_s);
  for (const entry of state.videoInfo.values()) setVideoLocal(entry.video, local);
  updateAtTime(local);
}

function applyFrame(frameIndex) {
  if (!state.transforms || !state.episode) return;
  const frame = Math.min(frameIndex, state.episode.frame_count - 1);
  if (frame === state.currentFrame) return;
  const cfg = configState();
  const sideCount = Array.isArray(cfg.sides) ? cfg.sides.length : 1;
  const meshCount = cfg.meshCount;
  const frameStride = sideCount * meshCount * 16;
  const rowStride = meshCount * 16;
  for (let side = 0; side < sideCount; side += 1) {
    for (let meshIndex = 0; meshIndex < meshCount; meshIndex += 1) {
      const offset = frame * frameStride + side * rowStride + meshIndex * 16;
      const target = sideCount > 1 ? state.meshes[side]?.[meshIndex] : state.meshes[meshIndex];
      if (!target) continue;
      target.matrix.fromArray(state.transforms, offset);
      target.matrixWorldNeedsUpdate = true;
    }
  }
  applyPhaseVisual(frame);
  state.currentFrame = frame;
  elements.frame.textContent = "frame " + String(frame).padStart(4, "0") + " / " + (state.episode.frame_count - 1);
}

function applyPhaseVisual(frame) {
  const timeline = state.episode.phase_timeline;
  elements.phase.hidden = true;
  if (!timeline || !timeline.has_gate_records) return;
  const phase = timeline.phase_ids[frame] === 1 ? "async" : "sync";
  elements.phase.hidden = false;
  elements.phase.textContent = phase.toUpperCase() + " · step " + frame;
  const colors = PHASE_COLORS[phase];
  if (colors && state.materials.length >= 2) {
    state.materials[0].color.setHex(colors[0]);
    state.materials[1].color.setHex(colors[1]);
  }
}

async function loadEpisode() {
  const generation = ++state.loadGeneration;
  pauseAll();
  state.transforms = null;
  state.currentFrame = -1;
  elements.play.disabled = true;
  elements.status.textContent = "Loading…";

  const dataset = elements.dataset.value;
  const episodeIndex = Number(elements.episode.value);
  const episode = await fetchChecked("/api/episode/" + encodeURIComponent(dataset) + "/" + episodeIndex);
  if (generation !== state.loadGeneration) return;

  state.episode = episode;
  setupCameraGrid(episode);
  elements.timeline.max = String(episode.duration_s);
  elements.timeline.step = String(1 / episode.fps);
  elements.timeline.value = "0";
  const signalText = state.signal ? " · " + state.signal : "";
  const sidText = state.sidRecoveryMetrics
    ? " · SID " + state.sidRecoveryCount + " samples · mean Δd "
      + (100 * state.sidRecoveryMetrics.mean_nearest_manifold_distance_reduction_fraction).toFixed(1) + "%"
    : "";
  elements.meta.textContent = dataset + " · episode " + episodeIndex + signalText + " · " + episode.frame_count + " frames" + sidText;
  elements.trajectoryLabel.textContent = state.sidRecoveryMetrics ? "SID recovery at EE TCP" : "3D" + signalText;

  const videoPromises = Promise.all(Array.from(state.videos.values()).map(waitForMetadata));
  const [buffer] = await Promise.all([
    fetchChecked(transformsUrl(episode), "arrayBuffer"),
    videoPromises,
  ]);
  if (generation !== state.loadGeneration) return;

  const cfg = configState();
  const sideCount = Array.isArray(cfg.sides) ? cfg.sides.length : 1;
  const expected = episode.frame_count * sideCount * cfg.meshCount * 16;
  if (buffer.byteLength !== expected * Float32Array.BYTES_PER_ELEMENT) {
    throw new Error("transform size mismatch: " + buffer.byteLength + " bytes, expected " + (expected * Float32Array.BYTES_PER_ELEMENT));
  }
  state.transforms = new Float32Array(buffer);
  await loadEETrajectory(episode, generation);
  if (generation !== state.loadGeneration) return;
  setLocalTime(0);
  elements.play.disabled = false;
  elements.status.textContent = "Synced";
}

function pauseAll() {
  for (const video of state.videos.values()) video.pause();
  elements.play.textContent = "▶";
  elements.play.setAttribute("aria-label", "play");
}

async function playAll() {
  if (!state.episode || !state.transforms) return;
  if (localTime() >= state.episode.duration_s - 1 / state.episode.fps) setLocalTime(0);
  const rate = Number(elements.rate.value);
  for (const video of state.videos.values()) video.playbackRate = rate;
  await Promise.all(Array.from(state.videos.values()).map((video) => video.play()));
  elements.play.textContent = "II";
  elements.play.setAttribute("aria-label", "pause");
  elements.status.textContent = "Playing";
}

function togglePlayback() {
  const master = state.videos.get(state.masterCamera);
  if (!master) return;
  if (master.paused) playAll().catch(showError);
  else {
    pauseAll();
    elements.status.textContent = "Paused";
  }
}

function synchronizeVideos(local) {
  const master = state.videos.get(state.masterCamera);
  if (!master || master.paused) return;
  for (const entry of state.videoInfo.values()) {
    if (entry.video === master) continue;
    if (Math.abs(entry.video.currentTime - (entry.fromS + local)) > 0.08) entry.video.currentTime = entry.fromS + local;
  }
}

function tick() {
  controls.update();
  if (state.episode && state.transforms && !state.scrubbing) {
    const local = localTime();
    updateAtTime(local);
    synchronizeVideos(local);
    const master = state.videos.get(state.masterCamera);
    if (master && !master.paused && local >= state.episode.duration_s - 1 / state.episode.fps) {
      setLocalTime(state.episode.duration_s);
      pauseAll();
      elements.status.textContent = "Episode ended";
    }
  }
  renderer.render(scene, camera);
}

function showError(error) {
  pauseAll();
  elements.status.textContent = "Load failed";
  console.error(error);
  alert(error.message);
}

elements.dataset.addEventListener("change", () => {
  populateEpisodes();
  loadEpisode().catch(showError);
});
elements.episode.addEventListener("change", () => loadEpisode().catch(showError));
elements.signal.addEventListener("change", () => {
  state.signal = elements.signal.value;
  loadEpisode().catch(showError);
});
elements.refreshDatasets.addEventListener("click", async () => {
  const selected = elements.dataset.value;
  pauseAll();
  elements.status.textContent = "Scanning…";
  try {
    state.config = await fetchChecked("/api/config?ts=" + Date.now());
    populateDatasets(selected);
    populateEpisodes();
    await setupSIDRecoveryOverlay(state.config);
    await loadEpisode();
  } catch (error) {
    showError(error);
  }
});
elements.resetView.addEventListener("click", resetCamera);
elements.play.addEventListener("click", togglePlayback);
for (const video of document.querySelectorAll("video")) video?.addEventListener("click", togglePlayback);
elements.rate.addEventListener("change", () => {
  for (const video of state.videos.values()) video.playbackRate = Number(elements.rate.value);
});
elements.timeline.addEventListener("pointerdown", () => { state.scrubbing = true; });
elements.timeline.addEventListener("input", () => setLocalTime(Number(elements.timeline.value)));
elements.timeline.addEventListener("change", () => {
  state.scrubbing = false;
  setLocalTime(Number(elements.timeline.value));
});
document.addEventListener("keydown", (event) => {
  if (event.code === "Space" && event.target.tagName !== "SELECT") {
    event.preventDefault();
    togglePlayback();
  }
});

async function initialize() {
  state.config = await fetchChecked("/api/config");
  elements.title.textContent = state.config.title || "robo-visualize";
  elements.eyebrow.textContent = (state.config.title || "robo-visualize").toUpperCase();
  elements.refreshDatasets.hidden = !state.config.datasetScanner;
  populateSignal();
  populateDatasets();
  populateEpisodes();
  await loadRobotMeshes();
  setupPerturbationOverlay(state.config);
  await setupSIDRecoveryOverlay(state.config);
  await loadEpisode();
  if (state.config.live_mode) startLivePolling();
  renderer.setAnimationLoop(tick);
}

initialize().catch(showError);
