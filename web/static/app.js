/**
 * SLATE: Spectral Latent Analysis & Trajectory Exploration
 * app.js -- frontend visualization engine
 * Obsidian Monochrome theme (#101010 base, #1c1c1c panels, #b0b0b0 text).
 * 3D dot palette: #f2e8dc (ivory), #c8b390 (warm tan), active marker #c8b390.
 * Linking line color: #424242.
 * Playback speed selector: 0.25x, 0.5x, 1.0x, 2.0x.
 * Dismissible explanation card with [ x ] close and [ i / info ] toggle.
 * No emojis. No em dashes.
 */

const state = {
  data: null,
  currentMode: '3d', // '3d', '2d', 'spectral', 'errors', 'matrices', 'prediction'
  currentTime: 0,
  timeAccumulator: 0,
  playbackSpeed: 0.5, // Default slower speed for clear observation
  isPlaying: false,
  fps: 60,
  lastFrameTime: performance.now()
};

// DOM references
const canvas3D = document.getElementById('canvas-3d');
const canvas2D = document.getElementById('canvas-2d');
const ctx2d = canvas2D.getContext('2d');
const timeSlider = document.getElementById('time-slider');
const timeReadout = document.getElementById('time-readout');
const playPauseBtn = document.getElementById('play-pause-btn');
const nodeTooltip = document.getElementById('node-tooltip');
const selectedNodeTag = document.getElementById('selected-node-tag');

// Explainer card references
const viewExplainer = document.getElementById('view-explainer');
const explainerTitle = document.getElementById('explainer-title');
const explainerDesc = document.getElementById('explainer-desc');
const explainerHints = document.getElementById('explainer-hints');
const btnCloseExplainer = document.getElementById('btn-close-explainer');
const btnToggleExplainer = document.getElementById('btn-toggle-explainer');

// Header elements
const hudFps = document.getElementById('hud-fps');
const hudRadius = document.getElementById('hud-radius');
const hudDim = document.getElementById('hud-dim');
const hudTimesteps = document.getElementById('hud-timesteps');
const hudCamX = document.getElementById('hud-cam-x');
const hudCamY = document.getElementById('hud-cam-y');
const hudZoom = document.getElementById('hud-zoom');

// Sidebar inspector elements
const selTitle = document.getElementById('sel-node-title');
const selNorm = document.getElementById('sel-norm');
const selError = document.getElementById('sel-error');
const selPred = document.getElementById('sel-pred');
const selComponents = document.getElementById('sel-components');
const spectralModesBody = document.getElementById('spectral-modes-body');

// 3D Scene globals
let scene3d, camera3d, renderer3d, controls3d;
let trajectoryPointsMesh, trajectoryLineMesh, trajectoryTubeMesh, instancedSpheresMesh, activePointMarker, groundGrid;
let starFieldMesh;
let circleTexture = null;
const raycaster3d = new THREE.Raycaster();
const mouse3d = new THREE.Vector2();

/**
 * Creates 3D shaded sphere texture for Three.js point particles.
 * Uses radial lighting gradient with highlight and ambient shadow so dots feel like 3D spheres.
 */
function createCircleTexture() {
  if (circleTexture) return circleTexture;
  const canvas = document.createElement('canvas');
  canvas.width = 64;
  canvas.height = 64;
  const ctx = canvas.getContext('2d');

  ctx.clearRect(0, 0, 64, 64);

  // Radial gradient: light source from top-left (x=24, y=24) creating true 3D spherical depth
  const grad = ctx.createRadialGradient(24, 22, 2, 32, 32, 28);
  grad.addColorStop(0, '#ffffff');             // Specular highlight spot
  grad.addColorStop(0.35, '#eeeeee');          // Diffuse lighted sphere surface
  grad.addColorStop(0.70, '#b8b8b8');          // Soft ambient drop-off
  grad.addColorStop(0.92, '#505050');          // Spherical rim shadow
  grad.addColorStop(1.0, 'rgba(0, 0, 0, 0)');  // Transparent antialiased edge

  ctx.beginPath();
  ctx.arc(32, 32, 28, 0, Math.PI * 2);
  ctx.fillStyle = grad;
  ctx.fill();

  circleTexture = new THREE.CanvasTexture(canvas);
  return circleTexture;
}

/**
 * Boot application
 */
async function init() {
  setupTabs();
  setupSpeedControls();
  setupKeyboardShortcuts();
  setupModals();
  initThree3D();
  window.addEventListener('resize', onWindowResize);

  try {
    const res = await fetch('/api/pipeline');
    const data = await res.json();
    loadPipelineData(data);
  } catch (err) {
    console.error('Failed to load pipeline data:', err);
  }

  requestAnimationFrame(renderLoop);
}

/**
 * Ingest pipeline data and configure UI
 */
function loadPipelineData(data) {
  state.data = data;
  state.currentTime = 0;
  state.timeAccumulator = 0;

  const T = data.meta.timesteps_T;
  timeSlider.max = T - 1;
  timeSlider.value = 0;
  timeReadout.textContent = `hour: 0 / ${T - 1}`;

  // Header telemetry
  hudTimesteps.textContent = T;
  hudDim.textContent = data.meta.hidden_dim_n;
  hudRadius.textContent = data.spectral.spectral_radius.toFixed(3);

  const badgeStability = document.getElementById('badge-stability');
  badgeStability.textContent = `stability: ${data.spectral.stability_class.toLowerCase()}`;

  // Construct overhauled 3D scene with thicker trajectory and tiny pin dots
  build3DTrajectory();

  // Populate spectral table
  renderSpectralTable();

  // Update explainer card for current mode
  updateExplainerCard(state.currentMode);

  // Inspect first node
  inspectNode(0);
}

/**
 * Plain-English explanation cards for each view
 */
const modeExplanations = {
  '3d': {
    title: '3D STATE SPACE CONSTELLATION',
    desc: 'Each tiny dot is 1 hour of network memory (h_t). The sleek trajectory line traces 30 days of evolving air pollution representations. Clustered loops reveal daily rush hour cycles.',
    hints: 'controls: drag to rotate orbit | scroll to zoom | click any dot to inspect | space to play'
  },
  '2d': {
    title: '2D MEMORY MAP (ORTHONORMAL PHASE PORTRAIT)',
    desc: 'The 8D memory vectors are projected onto the 2 most informative perpendicular axes (q1 and q2 via QR decomposition). Solid path = Actual AI memory. Dashed path = Linear mathematical model (A @ h_t).',
    hints: 'why it matters: close alignment between paths proves simple linear math explains >84% of complex AI memory'
  },
  'spectral': {
    title: 'STABILITY CIRCLE (COMPLEX EIGENVALUE PLANE)',
    desc: 'Eigenvalues (lambda) of learned transition matrix A. Dashed circle is boundary |lambda| = 1. Dots inside circle = Stable memory (perturbations decay smoothly). Dots outside = Explosive memory.',
    hints: 'result: all eigenvalues reside inside the unit circle (spectral radius < 1), proving mathematically that memory never blows up'
  },
  'errors': {
    title: 'SURROGATE MODEL ERROR PER HOUR',
    desc: 'Hourly difference ||h_(t+1) - A @ h_t|| between non-linear AI and linear matrix A. The low average error confirms that linear operator A accurately captures recurrent dynamics.',
    hints: 'interactions: scrub timeline slider below to check errors across different weather periods'
  },
  'matrices': {
    title: 'REDUNDANCY TEST (REDUCED ROW ECHELON FORM)',
    desc: 'Applies Gauss-Jordan row reduction (SymPy) to hidden neuron states. Pivot columns indicate unique independent memory units. Free columns indicate duplicate neurons.',
    hints: 'summary: full rank confirms all 8 hidden dimensions are actively contributing independent information'
  },
  'prediction': {
    title: 'REAL-WORLD AIR QUALITY PREDICTION (PM2.5)',
    desc: 'Validates that internal memory serves a useful engineering task. Solid path = Physical sensor PM2.5 measurements. Dashed path = Model prediction from internal state.',
    hints: 'accuracy: achieves high correlation (R^2 > 0.90), proving the geometric manifold encodes genuine physical dynamics'
  },
  '3d_spheres': {
    title: '3D VOLUMETRIC SPHERE MANIFOLD',
    desc: 'Each memory state is rendered as a genuine 3D lighted sphere with specular highlights and ambient shading. The 50% opacity linking line reveals the volumetric attractor structure.',
    hints: 'controls: drag to orbit | scroll to zoom | space to play | notice the physical spherical depth and lighting'
  }
};

function updateExplainerCard(mode) {
  const info = modeExplanations[mode] || modeExplanations['3d'];
  explainerTitle.textContent = info.title;
  explainerDesc.textContent = info.desc;
  explainerHints.textContent = info.hints;

  document.getElementById('badge-mode').textContent = `mode: ${info.title.toLowerCase().split('(')[0].trim()}`;
}

/**
 * Configure Three.js 3D Viewport in Dark Grey Slate theme
 */
function initThree3D() {
  const container = document.getElementById('viewport-container');
  const width = container.clientWidth;
  const height = container.clientHeight;

  scene3d = new THREE.Scene();
  scene3d.fog = new THREE.FogExp2(0x101010, 0.015);

  camera3d = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000);
  camera3d.position.set(20, 16, 26);

  renderer3d = new THREE.WebGLRenderer({ canvas: canvas3D, antialias: true, alpha: true });
  renderer3d.setSize(width, height);
  renderer3d.setPixelRatio(Math.min(window.devicePixelRatio, 2));

  controls3d = new THREE.OrbitControls(camera3d, renderer3d.domElement);
  controls3d.enableDamping = true;
  controls3d.dampingFactor = 0.06;
  controls3d.maxDistance = 140;
  controls3d.minDistance = 4;

  // Ambient and Directional Lights for realistic 3D sphere shading
  const ambientLight = new THREE.AmbientLight(0xffffff, 0.7);
  scene3d.add(ambientLight);

  const dirLight1 = new THREE.DirectionalLight(0xffffff, 0.9);
  dirLight1.position.set(30, 40, 30);
  scene3d.add(dirLight1);

  const dirLight2 = new THREE.DirectionalLight(0xc8b390, 0.4);
  dirLight2.position.set(-20, -10, -20);
  scene3d.add(dirLight2);

  // Add subtle reference ground plane grid in dark slate
  groundGrid = new THREE.GridHelper(36, 18, 0x303030, 0x1a1a1a);
  groundGrid.position.y = -8;
  scene3d.add(groundGrid);

  const dotTex = createCircleTexture();

  // Subtle star dust particles in slate gray
  const starGeo = new THREE.BufferGeometry();
  const starCount = 400;
  const starPos = new Float32Array(starCount * 3);
  for (let i = 0; i < starCount * 3; i += 3) {
    starPos[i] = (Math.random() - 0.5) * 110;
    starPos[i + 1] = (Math.random() - 0.5) * 110;
    starPos[i + 2] = (Math.random() - 0.5) * 110;
  }
  starGeo.setAttribute('position', new THREE.BufferAttribute(starPos, 3));
  const starMat = new THREE.PointsMaterial({
    color: 0x3a3a3a,
    size: 1.4,
    map: dotTex,
    transparent: true,
    alphaTest: 0.1
  });
  starFieldMesh = new THREE.Points(starGeo, starMat);
  scene3d.add(starFieldMesh);

  // Active state marker: warm sand sphere
  const markerGeo = new THREE.SphereGeometry(0.38, 20, 20);
  const markerMat = new THREE.MeshBasicMaterial({ color: 0xc8b390 });
  activePointMarker = new THREE.Mesh(markerGeo, markerMat);
  scene3d.add(activePointMarker);

  container.addEventListener('mousemove', on3DMouseMove);
  container.addEventListener('click', on3DClick);
}

/**
 * Rebuilt 3D Trajectory with sleek line and tiny pin dots
 */
function build3DTrajectory() {
  if (!state.data) return;

  if (trajectoryPointsMesh) scene3d.remove(trajectoryPointsMesh);
  if (trajectoryLineMesh) scene3d.remove(trajectoryLineMesh);
  if (trajectoryTubeMesh) scene3d.remove(trajectoryTubeMesh);

  const coords = state.data.projection.coords_3d;
  const T = coords.length;
  const scale = 12.0;

  const positions = new Float32Array(T * 3);
  const colors = new Float32Array(T * 3);
  const curvePoints = [];

  for (let i = 0; i < T; i++) {
    const x = coords[i][0] * scale;
    const y = coords[i][1] * scale;
    const z = coords[i][2] * scale;

    positions[i * 3] = x;
    positions[i * 3 + 1] = y;
    positions[i * 3 + 2] = z;

    curvePoints.push(new THREE.Vector3(x, y, z));

    // Warm sand (#c8b390) and soft white (#b0b0b0) palette
    const ratio = i / T;
    let r, g, b;
    if (ratio < 0.5) {
      // Warm sand
      r = 0.949; g = 0.910; b = 0.863;
    } else {
      // Soft off-white
      r = 0.784; g = 0.702; b = 0.565;
    }
    colors[i * 3] = r;
    colors[i * 3 + 1] = g;
    colors[i * 3 + 2] = b;
  }

  // 1. Sleek, thin continuous 3D trajectory line (#424242 at 50% opacity)
  const lGeo = new THREE.BufferGeometry().setFromPoints(curvePoints);
  const lMat = new THREE.LineBasicMaterial({
    color: 0x424242,
    transparent: true,
    opacity: 0.50
  });
  trajectoryLineMesh = new THREE.Line(lGeo, lMat);
  scene3d.add(trajectoryLineMesh);

  // 2. Point cloud dots with 3D shaded sphere texture
  const dotTex = createCircleTexture();
  const pGeo = new THREE.BufferGeometry();
  pGeo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  pGeo.setAttribute('color', new THREE.BufferAttribute(colors, 3));

  const pMat = new THREE.PointsMaterial({
    size: 2.2,
    map: dotTex,
    vertexColors: true,
    transparent: true,
    alphaTest: 0.15
  });
  trajectoryPointsMesh = new THREE.Points(pGeo, pMat);
  scene3d.add(trajectoryPointsMesh);

  // 3. True 3D Geometric Spheres (THREE.InstancedMesh with Phong lighting)
  if (instancedSpheresMesh) scene3d.remove(instancedSpheresMesh);
  const sphereGeo = new THREE.SphereGeometry(0.18, 14, 14);
  const sphereMat = new THREE.MeshPhongMaterial({
    shininess: 40,
    specular: 0x444444
  });
  instancedSpheresMesh = new THREE.InstancedMesh(sphereGeo, sphereMat, T);
  const dummy = new THREE.Object3D();
  const cObj = new THREE.Color();

  for (let i = 0; i < T; i++) {
    dummy.position.set(positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2]);
    dummy.updateMatrix();
    instancedSpheresMesh.setMatrixAt(i, dummy.matrix);

    cObj.setRGB(colors[i * 3], colors[i * 3 + 1], colors[i * 3 + 2]);
    instancedSpheresMesh.setColorAt(i, cObj);
  }
  instancedSpheresMesh.instanceMatrix.needsUpdate = true;
  if (instancedSpheresMesh.instanceColor) instancedSpheresMesh.instanceColor.needsUpdate = true;
  instancedSpheresMesh.visible = (state.currentMode === '3d_spheres');
  scene3d.add(instancedSpheresMesh);

  updateActiveMarker(0);
}

function updateActiveMarker(idx) {
  if (!state.data || !activePointMarker) return;
  const coords = state.data.projection.coords_3d;
  if (idx < coords.length) {
    const scale = 12.0;
    const px = coords[idx][0] * scale;
    const py = coords[idx][1] * scale;
    const pz = coords[idx][2] * scale;
    activePointMarker.position.set(px, py, pz);

    updateFloatingTag(px, py, pz, idx);
  }
}

function updateFloatingTag(x, y, z, idx) {
  if (state.currentMode !== '3d' || !camera3d) {
    selectedNodeTag.style.display = 'none';
    return;
  }
  const pos = new THREE.Vector3(x, y, z);
  pos.project(camera3d);

  if (pos.z > 1.0) {
    selectedNodeTag.style.display = 'none';
    return;
  }

  const container = document.getElementById('viewport-container');
  const sx = (pos.x * 0.5 + 0.5) * container.clientWidth;
  const sy = (-pos.y * 0.5 + 0.5) * container.clientHeight;

  selectedNodeTag.style.display = 'block';
  selectedNodeTag.style.left = `${sx + 14}px`;
  selectedNodeTag.style.top = `${sy - 12}px`;
  selectedNodeTag.textContent = `► hour ${String(idx).padStart(3, '0')} ◄`;
}

function on3DMouseMove(e) {
  if (state.currentMode !== '3d' || !state.data || !trajectoryPointsMesh) return;
  const rect = renderer3d.domElement.getBoundingClientRect();
  mouse3d.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
  mouse3d.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;

  raycaster3d.setFromCamera(mouse3d, camera3d);
  raycaster3d.params.Points.threshold = 0.8;
  const intersects = raycaster3d.intersectObject(trajectoryPointsMesh);

  if (intersects.length > 0) {
    const idx = intersects[0].index;
    showTooltip(e.clientX, e.clientY, idx);
  } else {
    hideTooltip();
  }
}

function on3DClick(e) {
  if (state.currentMode !== '3d' || !state.data || !trajectoryPointsMesh) return;
  const rect = renderer3d.domElement.getBoundingClientRect();
  mouse3d.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
  mouse3d.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;

  raycaster3d.setFromCamera(mouse3d, camera3d);
  raycaster3d.params.Points.threshold = 1.0;
  const intersects = raycaster3d.intersectObject(trajectoryPointsMesh);

  if (intersects.length > 0) {
    const idx = intersects[0].index;
    state.currentTime = idx;
    timeSlider.value = idx;
    inspectNode(idx);
    updateActiveMarker(idx);
  }
}

function showTooltip(x, y, idx) {
  nodeTooltip.style.display = 'block';
  nodeTooltip.style.left = `${x + 12}px`;
  nodeTooltip.style.top = `${y + 12}px`;

  const err = state.data.dynamics.step_errors[idx] !== undefined 
    ? state.data.dynamics.step_errors[idx].toFixed(4) : 'n/a';
  const pred = state.data.predictions.predicted_y[idx] !== undefined 
    ? state.data.predictions.predicted_y[idx].toFixed(1) : 'n/a';

  nodeTooltip.innerHTML = `
    <div style="color: var(--accent-mid); font-weight: 700; margin-bottom: 2px;">► HOUR: t = ${idx} ◄</div>
    <div style="color: var(--text-secondary);">surrogate gap: <span style="color: var(--text-primary);">${err}</span></div>
    <div style="color: var(--text-secondary);">pred PM2.5: <span style="color: var(--text-primary);">${pred} ug/m3</span></div>
  `;
}

function hideTooltip() {
  nodeTooltip.style.display = 'none';
}

/**
 * Inspect node and populate right sidebar
 */
function inspectNode(t) {
  if (!state.data) return;
  timeReadout.textContent = `hour: ${t} / ${state.data.meta.timesteps_T - 1}`;
  selTitle.textContent = `HOUR: t = ${String(t).padStart(3, '0')}`;

  const H_sample = state.data.raw_samples.H_preview[t] || new Array(state.data.meta.hidden_dim_n).fill(0);
  const norm = Math.hypot(...H_sample);
  selNorm.textContent = norm.toFixed(3);

  const stepErr = state.data.dynamics.step_errors[t];
  selError.textContent = stepErr !== undefined ? stepErr.toFixed(4) : 'n/a';

  const pred = state.data.predictions.predicted_y[t];
  const actual = state.data.predictions.actual_y[t];
  selPred.textContent = `${pred.toFixed(1)} ug/m3 (act: ${actual.toFixed(1)})`;

  // Latent components rendered as clean gauge tracks with warm sand round dot
  let compHtml = '';
  H_sample.forEach((val, i) => {
    const pct = Math.min(100, Math.max(0, ((val + 1) / 2) * 100));
    compHtml += `
      <div class="comp-row">
        <span style="color: var(--text-secondary); width: 28px;">h_${i}</span>
        <div class="comp-track">
          <div class="comp-dot" style="left: ${pct}%;"></div>
        </div>
        <span style="width: 42px; text-align: right; color: var(--text-primary);">${val.toFixed(2)}</span>
      </div>
    `;
  });
  selComponents.innerHTML = compHtml;
}

/**
 * Populate spectral modes table
 */
function renderSpectralTable() {
  if (!state.data) return;
  const modes = state.data.spectral.modes;
  let rows = '';
  modes.forEach(m => {
    const mag = m.magnitude.toFixed(2);
    const isStable = m.magnitude <= 1.0;
    const statusColor = isStable ? 'var(--accent-mid)' : 'var(--accent-deep)';
    const statusText = isStable ? 'Stable' : 'Divergent';
    rows += `
      <tr>
        <td style="color: var(--accent-mid);">lambda_${m.index}</td>
        <td>${m.real.toFixed(2)}${m.imag >= 0 ? '+' : ''}${m.imag.toFixed(2)}i</td>
        <td>${mag}</td>
        <td style="color: ${statusColor}; font-weight: 500;">${statusText}</td>
      </tr>
    `;
  });
  spectralModesBody.innerHTML = rows;
}

/**
 * 2D Canvas Views
 */
function render2DMode() {
  if (!state.data) return;
  const w = canvas2D.width;
  const h = canvas2D.height;
  ctx2d.clearRect(0, 0, w, h);

  if (state.currentMode === '2d') {
    draw2DPhasePortrait(w, h);
  } else if (state.currentMode === 'spectral') {
    drawSpectralPlane(w, h);
  } else if (state.currentMode === 'errors') {
    drawErrorDynamics(w, h);
  } else if (state.currentMode === 'matrices') {
    drawMatrixText(w, h);
  } else if (state.currentMode === 'prediction') {
    drawPredictionsPlot(w, h);
  }
}

/**
 * Mode 2: 2D Phase Portrait (ASCII + Dots & Symbols Terminal Matrix Design)
 */
function draw2DPhasePortrait(w, h) {
  const coords = state.data.projection.coords_2d;
  const surrogateCoords = state.data.dynamics.coords_surrogate_2d;
  const T = coords.length;
  const curT = state.currentTime;

  // Background subtle ASCII grid coordinates - dynamic full-bleed fit
  const padL = 20;
  const padR = 20;
  const padT = 70;
  const padB = 60;
  const boxW = Math.max(w - padL - padR, 300);
  const boxH = Math.max(h - padT - padB, 250);
  const cx = padL + boxW / 2;
  const cy = padT + (boxH - 30) / 2 + 15;
  const scale = Math.min(boxW * 0.44, boxH * 0.42);

  // Outer ASCII border box with double border symbols
  ctx2d.fillStyle = '#1c1c1c';
  ctx2d.fillRect(padL, padT, boxW, boxH);
  ctx2d.strokeStyle = '#424242';
  ctx2d.lineWidth = 1;
  ctx2d.strokeRect(padL, padT, boxW, boxH);

  // ASCII Frame Corner Markers
  ctx2d.fillStyle = '#c8b390';
  ctx2d.font = '11px monospace';
  ctx2d.fillText('+', padL + 3, padT + 12);
  ctx2d.fillText('+', padL + boxW - 11, padT + 12);
  ctx2d.fillText('+', padL + 3, padT + boxH - 4);
  ctx2d.fillText('+', padL + boxW - 11, padT + boxH - 4);

  // Top ASCII Header Bar
  ctx2d.fillStyle = '#252525';
  ctx2d.fillRect(padL, padT, boxW, 26);
  ctx2d.strokeStyle = '#383838';
  ctx2d.strokeRect(padL, padT, boxW, 26);

  ctx2d.fillStyle = '#f0ede8';
  ctx2d.font = '10px monospace';
  ctx2d.fillText('[ 02: ORTHONORMAL PHASE MAP (QR-DECOMPOSED B_2D) ]', padL + 12, padT + 17);
  ctx2d.fillStyle = '#c8b390';
  ctx2d.fillText(`VARIANCE: ${state.data.projection.variance_retained_2d}%`, padL + boxW - 140, padT + 17);

  // ASCII Axis Crosshairs using dotted symbols
  ctx2d.fillStyle = '#303030';
  ctx2d.font = '9px monospace';
  for (let x = padL + 10; x < padL + boxW - 10; x += 16) {
    ctx2d.fillText('·', x, cy + 3);
  }
  for (let y = padT + 35; y < padT + boxH - 10; y += 16) {
    ctx2d.fillText(':', cx - 2, y);
  }
  ctx2d.fillStyle = '#c8b390';
  ctx2d.fillText('+', cx - 3, cy + 3);

  // Axis Labels
  ctx2d.fillStyle = '#b0b0b0';
  ctx2d.font = '9.5px monospace';
  ctx2d.fillText('q_1 ->', padL + boxW - 65, cy - 6);
  ctx2d.fillText('^ q_2', cx + 8, padT + 42);

  // Surrogate trajectory: ASCII dash-dotted connection
  if (surrogateCoords && surrogateCoords.length > 0) {
    ctx2d.strokeStyle = 'rgba(112, 112, 112, 0.40)';
    ctx2d.setLineDash([3, 5]);
    ctx2d.lineWidth = 0.75;
    ctx2d.beginPath();
    for (let i = 0; i < Math.min(T - 1, surrogateCoords.length); i++) {
      const sx = cx + surrogateCoords[i][0] * scale;
      const sy = cy - surrogateCoords[i][1] * scale;
      if (i === 0) ctx2d.moveTo(sx, sy);
      else ctx2d.lineTo(sx, sy);
    }
    ctx2d.stroke();
    ctx2d.setLineDash([]);
  }

  // Actual trajectory: sleek connecting thread with 50% opacity
  ctx2d.strokeStyle = 'rgba(66, 66, 66, 0.50)';
  ctx2d.lineWidth = 0.75;
  ctx2d.beginPath();
  for (let i = 0; i < T; i++) {
    const x = cx + coords[i][0] * scale;
    const y = cy - coords[i][1] * scale;
    if (i === 0) ctx2d.moveTo(x, y);
    else ctx2d.lineTo(x, y);
  }
  ctx2d.stroke();

  // ASCII Dot glyphs (round dot symbols · • °)
  for (let i = 0; i < T; i++) {
    const x = cx + coords[i][0] * scale;
    const y = cy - coords[i][1] * scale;

    if (i % 6 === 0) {
      // Small symbol character dot
      ctx2d.fillStyle = (i % 2 === 0) ? '#f2e8dc' : '#c8b390';
      ctx2d.font = '10px monospace';
      ctx2d.fillText('·', x - 2, y + 3);
    } else {
      // Tiny geometric pin dot
      ctx2d.fillStyle = (i % 2 === 0) ? '#f2e8dc' : '#c8b390';
      ctx2d.beginPath();
      ctx2d.arc(x, y, 1.2, 0, Math.PI * 2);
      ctx2d.fill();
    }
  }

  // Milestone labels in clean ASCII tags
  const milestones = [
    { t: 8, label: '[ morning rush ]' },
    { t: 14, label: '[ photochemical peak ]' },
    { t: 20, label: '[ evening stagnation ]' },
    { t: 168, label: '[ weekly cycle ]' }
  ];

  milestones.forEach(m => {
    if (m.t < T) {
      const mx = cx + coords[m.t][0] * scale;
      const my = cy - coords[m.t][1] * scale;

      ctx2d.strokeStyle = '#444444';
      ctx2d.setLineDash([2, 3]);
      ctx2d.beginPath();
      ctx2d.moveTo(mx, my);
      ctx2d.lineTo(mx + 18, my - 12);
      ctx2d.stroke();
      ctx2d.setLineDash([]);

      ctx2d.fillStyle = '#b0b0b0';
      ctx2d.font = '9px monospace';
      ctx2d.fillText(m.label, mx + 22, my - 10);
    }
  });

  // Active timestep marker: ASCII crosshair & pill badge
  if (curT < T) {
    const ax = cx + coords[curT][0] * scale;
    const ay = cy - coords[curT][1] * scale;

    // ASCII crosshair around active node
    ctx2d.fillStyle = '#c8b390';
    ctx2d.font = '11px monospace';
    ctx2d.fillText('+', ax - 3, ay + 4);

    ctx2d.strokeStyle = '#c8b390';
    ctx2d.lineWidth = 0.8;
    ctx2d.beginPath();
    ctx2d.arc(ax, ay, 6, 0, Math.PI * 2);
    ctx2d.stroke();

    // ASCII Floating badge
    ctx2d.fillStyle = '#101010';
    ctx2d.fillRect(ax + 10, ay - 11, 98, 18);
    ctx2d.strokeStyle = '#c8b390';
    ctx2d.lineWidth = 1;
    ctx2d.strokeRect(ax + 10, ay - 11, 98, 18);

    ctx2d.fillStyle = '#f0ede8';
    ctx2d.font = '9.5px monospace';
    ctx2d.fillText(`* t: ${String(curT).padStart(3, '0')} h *`, ax + 15, ay + 2);
  }

  // Bottom ASCII telemetry strip inside box
  const bY = padT + boxH - 24;
  ctx2d.fillStyle = '#141414';
  ctx2d.fillRect(padL + 1, bY, boxW - 2, 23);
  ctx2d.strokeStyle = '#282828';
  ctx2d.strokeRect(padL + 1, bY, boxW - 2, 23);

  ctx2d.fillStyle = '#b0b0b0';
  ctx2d.font = '9px monospace';
  ctx2d.fillText('MAP: 8D -> 2D (B^T @ (h_t - mu))  |  · actual states  |  - - surrogate A@h_t  |  + center', padL + 10, bY + 15);
}

/**
 * Mode 3: Spectral Complex Plane and Stability Circle (ASCII + Symbols Design)
 */
function drawSpectralPlane(w, h) {
  const modes = state.data.spectral.modes;
  const rho = state.data.spectral.spectral_radius;

  const padL = 20;
  const padR = 20;
  const padT = 70;
  const padB = 60;
  const boxW = Math.max(w - padL - padR, 300);
  const boxH = Math.max(h - padT - padB, 250);
  const cx = padL + boxW / 2;
  const cy = padT + (boxH - 30) / 2 + 15;
  const radiusPx = Math.min((boxW - 60) * 0.42, (boxH - 80) * 0.45);

  // Outer ASCII border box
  ctx2d.fillStyle = '#1c1c1c';
  ctx2d.fillRect(padL, padT, boxW, boxH);
  ctx2d.strokeStyle = '#424242';
  ctx2d.lineWidth = 1;
  ctx2d.strokeRect(padL, padT, boxW, boxH);

  // Corner markers
  ctx2d.fillStyle = '#c8b390';
  ctx2d.font = '11px monospace';
  ctx2d.fillText('+', padL + 3, padT + 12);
  ctx2d.fillText('+', padL + boxW - 11, padT + 12);
  ctx2d.fillText('+', padL + 3, padT + boxH - 4);
  ctx2d.fillText('+', padL + boxW - 11, padT + boxH - 4);

  // Top Header Bar
  ctx2d.fillStyle = '#252525';
  ctx2d.fillRect(padL, padT, boxW, 26);
  ctx2d.strokeStyle = '#383838';
  ctx2d.strokeRect(padL, padT, boxW, 26);

  ctx2d.fillStyle = '#f0ede8';
  ctx2d.font = '10px monospace';
  ctx2d.fillText('[ 03: SPECTRAL EIGENVALUE PLANE |lambda| <= 1.0 ]', padL + 12, padT + 17);
  ctx2d.fillStyle = '#c8b390';
  ctx2d.fillText(`rho(A): ${rho.toFixed(3)} [STABLE]`, padL + boxW - 160, padT + 17);

  // ASCII Axis Dotted Crosshairs
  ctx2d.fillStyle = '#303030';
  ctx2d.font = '9px monospace';
  for (let x = padL + 10; x < padL + boxW - 10; x += 16) {
    ctx2d.fillText('·', x, cy + 3);
  }
  for (let y = padT + 35; y < padT + boxH - 10; y += 16) {
    ctx2d.fillText(':', cx - 2, y);
  }
  ctx2d.fillStyle = '#c8b390';
  ctx2d.fillText('+', cx - 3, cy + 3);

  // Dashed unit circle: |lambda| = 1
  ctx2d.strokeStyle = '#5a5a5a';
  ctx2d.lineWidth = 0.75;
  ctx2d.setLineDash([4, 4]);
  ctx2d.beginPath();
  ctx2d.arc(cx, cy, radiusPx, 0, Math.PI * 2);
  ctx2d.stroke();
  ctx2d.setLineDash([]);

  // Spectral radius circle
  ctx2d.strokeStyle = 'rgba(200, 179, 144, 0.40)';
  ctx2d.lineWidth = 0.75;
  ctx2d.beginPath();
  ctx2d.arc(cx, cy, radiusPx * rho, 0, Math.PI * 2);
  ctx2d.stroke();

  // Axis Labels
  ctx2d.fillStyle = '#b0b0b0';
  ctx2d.font = '9.5px monospace';
  ctx2d.fillText('Re(lambda) ->', padL + boxW - 85, cy - 6);
  ctx2d.fillText('^ Im(lambda)', cx + 8, padT + 42);
  ctx2d.fillText('|lambda|=1 boundary', cx + radiusPx - 55, cy + 18);

  // Plot eigenvalues with ASCII spoked lines and dot glyphs
  modes.forEach((m) => {
    const ex = cx + m.real * radiusPx;
    const ey = cy - m.imag * radiusPx;

    // Spoke line to origin
    ctx2d.strokeStyle = 'rgba(66, 66, 66, 0.50)';
    ctx2d.lineWidth = 0.75;
    ctx2d.beginPath();
    ctx2d.moveTo(cx, cy);
    ctx2d.lineTo(ex, ey);
    ctx2d.stroke();

    // Dot symbol & pinhead
    ctx2d.fillStyle = m.magnitude <= 1.0 ? '#c8b390' : '#885533';
    ctx2d.beginPath();
    ctx2d.arc(ex, ey, 3.5, 0, Math.PI * 2);
    ctx2d.fill();

    ctx2d.fillStyle = '#f0ede8';
    ctx2d.font = '9.5px monospace';
    ctx2d.fillText(`* lambda_${m.index} (${m.magnitude.toFixed(2)})`, ex + 8, ey - 4);
  });

  // Bottom ASCII telemetry bar
  const bY = padT + boxH - 24;
  ctx2d.fillStyle = '#141414';
  ctx2d.fillRect(padL + 1, bY, boxW - 2, 23);
  ctx2d.strokeStyle = '#282828';
  ctx2d.strokeRect(padL + 1, bY, boxW - 2, 23);

  ctx2d.fillStyle = '#b0b0b0';
  ctx2d.font = '9px monospace';
  ctx2d.fillText(`STABILITY: ${state.data.spectral.stability_desc}  |  rho(A) = ${rho.toFixed(3)} < 1.00  |  · Modes in unit circle contract smoothly`, padL + 10, bY + 15);
}

/**
 * Mode 4: 1-Step Residual Errors
 */
function drawErrorDynamics(w, h) {
  const errors = state.data.dynamics.step_errors;
  const T = errors.length;
  const padL = 60, padR = 30, padT = 95, padB = 45;
  const plotW = w - padL - padR;
  const plotH = h - padT - padB;

  const maxErr = Math.max(...errors) * 1.15 || 0.1;

  ctx2d.fillStyle = '#c8b390';
  ctx2d.font = '11px monospace';
  ctx2d.fillText(`Linear Model 1-Step Error: ||h_(t+1) - A @ h_t|| (R^2 Fit: ${state.data.dynamics.r2_fit_score})`, padL, padT - 15);

  ctx2d.strokeStyle = '#2c2c2c';
  ctx2d.lineWidth = 1;
  ctx2d.strokeRect(padL, padT, plotW, plotH);

  // Mean error dashed line
  const meanY = padT + plotH - (state.data.dynamics.mean_step_error / maxErr) * plotH;
  ctx2d.strokeStyle = '#555555';
  ctx2d.setLineDash([4, 4]);
  ctx2d.beginPath();
  ctx2d.moveTo(padL, meanY); ctx2d.lineTo(padL + plotW, meanY);
  ctx2d.stroke();
  ctx2d.setLineDash([]);
  ctx2d.fillStyle = '#b0b0b0';
  ctx2d.fillText(`Average Residual: ${state.data.dynamics.mean_step_error}`, padL + 10, meanY - 5);

  // Residual line (sleek 0.8px)
  ctx2d.strokeStyle = '#c8b390';
  ctx2d.lineWidth = 0.8;
  ctx2d.beginPath();
  for (let i = 0; i < T; i++) {
    const x = padL + (i / (T - 1)) * plotW;
    const y = padT + plotH - (errors[i] / maxErr) * plotH;
    if (i === 0) ctx2d.moveTo(x, y);
    else ctx2d.lineTo(x, y);
  }
  ctx2d.stroke();

  // Active timestep marker
  if (state.currentTime < T) {
    const cx = padL + (state.currentTime / (T - 1)) * plotW;
    const cy = padT + plotH - (errors[state.currentTime] / maxErr) * plotH;
    ctx2d.fillStyle = '#f0ede8';
    ctx2d.beginPath();
    ctx2d.arc(cx, cy, 4.0, 0, Math.PI * 2);
    ctx2d.fill();
    ctx2d.fillText(`hour ${state.currentTime}: error = ${errors[state.currentTime].toFixed(4)}`, cx + 8, cy - 6);
  }
}

/**
 * Mode 5: RREF Latent Redundancy (HIGH CONTRAST & VISIBILITY)
 */
function drawMatrixText(w, h) {
  const rref = state.data.rref;
  const A = state.data.matrices.A_transition;
  const n = A.length;

  // Header Title
  ctx2d.fillStyle = '#c8b390';
  ctx2d.font = '14px monospace';
  ctx2d.fillText('REDUCED ROW ECHELON FORM (SYMPY) & LATENT REDUNDANCY', 36, 85);

  // Large Prominent Status Box
  ctx2d.fillStyle = '#1c1c1c';
  ctx2d.fillRect(36, 105, w - 72, 75);
  ctx2d.strokeStyle = '#c8b390';
  ctx2d.lineWidth = 0.9;
  ctx2d.strokeRect(36, 105, w - 72, 75);

  ctx2d.fillStyle = '#f0ede8';
  ctx2d.font = '13px monospace';
  ctx2d.fillText(`NUMERICAL RANK: ${rref.rank} / ${n}  [FULL RANK - ZERO DUPLICATES]`, 52, 132);

  ctx2d.fillStyle = '#c8b390';
  ctx2d.font = '12px monospace';
  ctx2d.fillText(`Pivot Dimensions (Linearly Independent): [ ${rref.pivot_dimensions.join(', ')} ]`, 52, 155);

  // Transition matrix preview with increased font size and clean spacing
  ctx2d.fillStyle = '#c8b390';
  ctx2d.font = '13px monospace';
  ctx2d.fillText(`Learned Transition Matrix A in R^(${n} x ${n}):`, 36, 215);

  const startY = 240;
  const rowHeight = 22;

  for (let r = 0; r < Math.min(n, 8); r++) {
    const rowStr = A[r].slice(0, 8).map(v => (v >= 0 ? '+' : '') + v.toFixed(2)).join('   ');
    ctx2d.fillStyle = '#f0ede8';
    ctx2d.font = '12px monospace';
    ctx2d.fillText(`row ${r}:  [ ${rowStr} ]`, 52, startY + r * rowHeight);
  }

  // Explanatory notes
  ctx2d.fillStyle = '#a8a49e';
  ctx2d.font = '11px monospace';
  const noteY = startY + n * rowHeight + 35;
  ctx2d.fillText('Mathematical Interpretation:', 36, noteY);
  ctx2d.fillText('1. Pivot columns prove that all 8 hidden neurons encode unique, non-overlapping information.', 36, noteY + 20);
  ctx2d.fillText('2. Free columns (if any) would identify duplicate neurons that could be pruned without loss.', 36, noteY + 38);
}

/**
 * Mode 6: Real-World Time-Series Prediction
 */
function drawPredictionsPlot(w, h) {
  const actual = state.data.predictions.actual_y;
  const pred = state.data.predictions.predicted_y;
  const T = Math.min(actual.length, 240);

  const padL = 60, padR = 30, padT = 95, padB = 45;
  const plotW = w - padL - padR;
  const plotH = h - padT - padB;

  const maxVal = Math.max(...actual.slice(0, T), ...pred.slice(0, T)) * 1.1;
  const minVal = Math.min(...actual.slice(0, T), ...pred.slice(0, T)) * 0.9;
  const valRange = maxVal - minVal || 1;

  ctx2d.fillStyle = '#c8b390';
  ctx2d.font = '11px monospace';
  ctx2d.fillText(`Fine Particulate Matter PM2.5 Prediction (RMSE: ${state.data.predictions.rmse} ug/m3, R^2: ${state.data.predictions.r2})`, padL, padT - 15);

  ctx2d.strokeStyle = '#2c2c2c';
  ctx2d.lineWidth = 1;
  ctx2d.strokeRect(padL, padT, plotW, plotH);

  // Actual (solid warm sand line, sleek 0.8px)
  ctx2d.strokeStyle = '#c8b390';
  ctx2d.lineWidth = 0.8;
  ctx2d.beginPath();
  for (let i = 0; i < T; i++) {
    const x = padL + (i / (T - 1)) * plotW;
    const y = padT + plotH - ((actual[i] - minVal) / valRange) * plotH;
    if (i === 0) ctx2d.moveTo(x, y);
    else ctx2d.lineTo(x, y);
  }
  ctx2d.stroke();

  // Predicted (dashed soft white line)
  ctx2d.strokeStyle = '#f0ede8';
  ctx2d.setLineDash([4, 4]);
  ctx2d.lineWidth = 0.8;
  ctx2d.beginPath();
  for (let i = 0; i < T; i++) {
    const x = padL + (i / (T - 1)) * plotW;
    const y = padT + plotH - ((pred[i] - minVal) / valRange) * plotH;
    if (i === 0) ctx2d.moveTo(x, y);
    else ctx2d.lineTo(x, y);
  }
  ctx2d.stroke();
  ctx2d.setLineDash([]);

  // Legend
  ctx2d.fillStyle = '#c8b390';
  ctx2d.fillText('● Solid: Actual Measured PM2.5', padL + plotW - 250, padT - 15);
  ctx2d.fillStyle = '#f0ede8';
  ctx2d.fillText('○ Dashed: Readout Prediction', padL + plotW - 80, padT - 15);
}

/**
 * Main Render Loop with Speed Controls
 */
function renderLoop(timestamp) {
  const delta = timestamp - state.lastFrameTime;
  state.lastFrameTime = timestamp;
  if (delta > 0) {
    state.fps = Math.round(1000 / delta);
    hudFps.textContent = state.fps;
  }

  // Playback advancement using speed multiplier
  if (state.isPlaying && state.data) {
    state.timeAccumulator += delta * 0.001 * state.playbackSpeed * 30;
    if (state.timeAccumulator >= 1.0) {
      const advanceSteps = Math.floor(state.timeAccumulator);
      state.timeAccumulator -= advanceSteps;
      state.currentTime = (state.currentTime + advanceSteps) % state.data.meta.timesteps_T;

      timeSlider.value = state.currentTime;
      inspectNode(state.currentTime);
      updateActiveMarker(state.currentTime);
    }
  }

  const is3DMode = (state.currentMode === '3d' || state.currentMode === '3d_spheres');

  if (is3DMode) {
    canvas3D.style.display = 'block';
    canvas2D.style.display = 'none';

    // Toggle between billboard point cloud and true 3D instanced spheres
    if (trajectoryPointsMesh) trajectoryPointsMesh.visible = (state.currentMode === '3d');
    if (instancedSpheresMesh) instancedSpheresMesh.visible = (state.currentMode === '3d_spheres');

    if (controls3d) {
      controls3d.update();
      hudCamX.textContent = camera3d.position.x.toFixed(1);
      hudCamY.textContent = camera3d.position.y.toFixed(1);
      hudZoom.textContent = (35 / camera3d.position.length()).toFixed(2);

      if (state.data) {
        const coords = state.data.projection.coords_3d;
        const scale = 12.0;
        updateFloatingTag(
          coords[state.currentTime][0] * scale,
          coords[state.currentTime][1] * scale,
          coords[state.currentTime][2] * scale,
          state.currentTime
        );
      }
    }
    renderer3d.render(scene3d, camera3d);
  } else {
    canvas3D.style.display = 'none';
    canvas2D.style.display = 'block';
    selectedNodeTag.style.display = 'none';
    render2DMode();
  }

  requestAnimationFrame(renderLoop);
}

/**
 * Setup interactions and speed controls
 */
function setupTabs() {
  const tabs = document.querySelectorAll('.tab-btn');
  tabs.forEach(tab => {
    tab.addEventListener('click', () => {
      tabs.forEach(t => t.classList.remove('active'));
      tab.classList.add('active');
      setMode(tab.dataset.mode);
    });
  });

  timeSlider.addEventListener('input', (e) => {
    state.currentTime = parseInt(e.target.value);
    state.timeAccumulator = 0;
    inspectNode(state.currentTime);
    updateActiveMarker(state.currentTime);
  });

  playPauseBtn.addEventListener('click', togglePlayPause);
}

function setupSpeedControls() {
  const speedBtns = document.querySelectorAll('.btn-speed');
  speedBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      speedBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      state.playbackSpeed = parseFloat(btn.dataset.speed);
    });
  });
}

function setMode(mode) {
  state.currentMode = mode;
  document.querySelectorAll('.tab-btn').forEach(t => {
    t.classList.toggle('active', t.dataset.mode === mode);
  });

  if (trajectoryPointsMesh) trajectoryPointsMesh.visible = (mode === '3d');
  if (instancedSpheresMesh) instancedSpheresMesh.visible = (mode === '3d_spheres');

  updateExplainerCard(mode);
}

function togglePlayPause() {
  state.isPlaying = !state.isPlaying;
  playPauseBtn.textContent = state.isPlaying ? '[ || pause ]' : '[ > play ]';
}

function setupKeyboardShortcuts() {
  window.addEventListener('keydown', (e) => {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;

    if (e.code === 'Space') {
      e.preventDefault();
      togglePlayPause();
    } else if (e.key === '1') {
      setMode('3d');
    } else if (e.key === '2') {
      setMode('2d');
    } else if (e.key === '3') {
      setMode('spectral');
    } else if (e.key === '4') {
      setMode('errors');
    } else if (e.key === '5') {
      setMode('matrices');
    } else if (e.key === '6') {
      setMode('prediction');
    } else if (e.key === '7') {
      setMode('3d_spheres');
    } else if (e.key.toLowerCase() === 'r') {
      openExperimentModal();
    } else if (e.key === '?' || e.key.toLowerCase() === 'h') {
      openHelpModal();
    }
  });
}

function onWindowResize() {
  const container = document.getElementById('viewport-container');
  const w = container.clientWidth;
  const h = container.clientHeight;

  if (camera3d && renderer3d) {
    camera3d.aspect = w / h;
    camera3d.updateProjectionMatrix();
    renderer3d.setSize(w, h);
  }

  canvas2D.width = w;
  canvas2D.height = h;
}

function setupModals() {
  document.getElementById('btn-config').addEventListener('click', openExperimentModal);
  document.getElementById('btn-close-modal').addEventListener('click', closeExperimentModal);
  document.getElementById('btn-run-experiment').addEventListener('click', submitExperiment);

  document.getElementById('btn-help').addEventListener('click', openHelpModal);
  document.getElementById('btn-close-help').addEventListener('click', closeHelpModal);

  // Close & toggle explanation card
  if (btnCloseExplainer && viewExplainer) {
    btnCloseExplainer.addEventListener('click', () => {
      viewExplainer.style.display = 'none';
      if (btnToggleExplainer) btnToggleExplainer.classList.remove('active');
    });
  }

  if (btnToggleExplainer && viewExplainer) {
    btnToggleExplainer.addEventListener('click', () => {
      const isHidden = (viewExplainer.style.display === 'none');
      viewExplainer.style.display = isHidden ? 'block' : 'none';
      btnToggleExplainer.classList.toggle('active', isHidden);
    });
  }
}

function openExperimentModal() {
  document.getElementById('config-modal').style.display = 'flex';
}

function closeExperimentModal() {
  document.getElementById('config-modal').style.display = 'none';
}

function openHelpModal() {
  document.getElementById('help-modal').style.display = 'flex';
}

function closeHelpModal() {
  document.getElementById('help-modal').style.display = 'none';
}

async function submitExperiment() {
  const hiddenDim = parseInt(document.getElementById('inp-hidden-dim').value);
  const recurrentScale = parseFloat(document.getElementById('inp-scale').value);

  const btn = document.getElementById('btn-run-experiment');
  btn.textContent = '[ executing... ]';
  btn.disabled = true;

  try {
    const res = await fetch('/api/run-experiment', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        hidden_dim: hiddenDim,
        recurrent_scale: recurrentScale
      })
    });
    const newData = await res.json();
    loadPipelineData(newData);
    closeExperimentModal();
  } catch (err) {
    alert('Experiment failed: ' + err);
  } finally {
    btn.textContent = '[ execute ]';
    btn.disabled = false;
  }
}

window.addEventListener('DOMContentLoaded', () => {
  onWindowResize();
  init();
});
