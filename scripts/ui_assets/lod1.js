"use strict";

function createLod1Viewer(canvas) {
  const gl = canvas.getContext("webgl", {antialias: true});
  if (!gl) throw new Error("当前浏览器无法使用 WebGL。可切回二维高度图。");

  const vertexSource = `
    attribute vec3 position;
    attribute float height;
    attribute float shade;
    uniform vec4 camera;
    uniform vec2 scale;
    uniform float depth;
    uniform float top;
    varying vec3 color;

    void main() {
      vec2 xy = vec2(position.x * camera.y - position.y * camera.x,
                     position.x * camera.x + position.y * camera.y);
      float z = position.z - top * 0.5;
      gl_Position = vec4(xy.x * scale.x,
                         (xy.y * camera.z + z * camera.w) * scale.y,
                         (xy.y * camera.w - z * camera.z) * depth, 1.0);
      float t = clamp(height / top, 0.0, 1.0);
      vec3 low = vec3(0.28, 0.49, 0.65);
      vec3 mid = vec3(0.34, 0.67, 0.54);
      vec3 high = vec3(0.98, 0.79, 0.45);
      vec3 roof = mix(mix(low, mid, min(1.0, 2.0 * t)), high, max(0.0, 2.0 * t - 1.0));
      color = height < 0.0 ? vec3(0.81, 0.86, 0.86) : roof * shade;
    }
  `;
  const fragmentSource = `
    precision mediump float;
    varying vec3 color;
    void main() { gl_FragColor = vec4(color, 1.0); }
  `;

  function compile(type, source) {
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      gl.deleteShader(shader);
      throw new Error("浏览器无法初始化 LOD1 渲染。");
    }
    return shader;
  }

  const vertex = compile(gl.VERTEX_SHADER, vertexSource);
  const fragment = compile(gl.FRAGMENT_SHADER, fragmentSource);
  const program = gl.createProgram();
  gl.attachShader(program, vertex);
  gl.attachShader(program, fragment);
  gl.linkProgram(program);
  gl.deleteShader(vertex);
  gl.deleteShader(fragment);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    gl.deleteProgram(program);
    throw new Error("浏览器无法初始化 LOD1 渲染。");
  }
  gl.useProgram(program);

  const buffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
  for (const [attribute, size, offset] of [["position", 3, 0], ["height", 1, 12], ["shade", 1, 16]]) {
    const slot = gl.getAttribLocation(program, attribute);
    gl.enableVertexAttribArray(slot);
    gl.vertexAttribPointer(slot, size, gl.FLOAT, false, 20, offset);
  }

  const camera = gl.getUniformLocation(program, "camera");
  const scale = gl.getUniformLocation(program, "scale");
  const depth = gl.getUniformLocation(program, "depth");
  const top = gl.getUniformLocation(program, "top");
  gl.clearColor(0.94, 0.96, 0.96, 1);
  gl.enable(gl.DEPTH_TEST);
  const defaultAzimuth = -0.57;
  const defaultPitch = 0.63;
  let meshCount = 0, gridCount = 0, radius = 1, maxHeight = 1;
  let frameWidth, frameHeight, azimuth, pitch, zoom, pointer;

  function draw() {
    const ratio = Math.min(2, window.devicePixelRatio || 1);
    const width = Math.round(canvas.clientWidth * ratio);
    const height = Math.round(canvas.clientHeight * ratio);
    if (!meshCount || !width || !height) return;
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }
    gl.viewport(0, 0, width, height);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    const aspect = width / height;
    const fit = 0.83 * zoom / Math.max(frameWidth / aspect, frameHeight);
    gl.uniform2f(scale, fit / aspect, fit);
    gl.uniform1f(depth, 0.92 / radius);
    gl.uniform1f(top, maxHeight);
    gl.uniform4f(camera, Math.sin(azimuth), Math.cos(azimuth), Math.sin(pitch), Math.cos(pitch));
    gl.drawArrays(gl.LINES, meshCount, gridCount);
    gl.drawArrays(gl.TRIANGLES, 0, meshCount);
  }

  function reset() {
    azimuth = defaultAzimuth;
    pitch = defaultPitch;
    zoom = 1;
    draw();
  }

  function zoomBy(factor) {
    zoom = Math.min(8, Math.max(0.5, zoom * factor));
    draw();
  }

  function rotate(dx, dy) {
    azimuth += dx * 0.008;
    pitch = Math.min(1.55, Math.max(0.15, pitch + dy * 0.008));
    draw();
  }

  function load(result) {
    const decoded = atob(result.mesh);
    const bytes = new Uint8Array(decoded.length);
    for (let i = 0; i < bytes.length; i++) bytes[i] = decoded.charCodeAt(i);
    const mesh = new Float32Array(bytes.buffer);
    maxHeight = result.height_max;
    const x = result.extent[0] * 0.53;
    const y = result.extent[1] * 0.53;
    radius = Math.hypot(x, y, maxHeight * 0.5);
    const sin = Math.abs(Math.sin(defaultAzimuth));
    const cos = Math.cos(defaultAzimuth);
    frameWidth = x * cos + y * sin;
    frameHeight = (x * sin + y * cos) * Math.sin(defaultPitch) + maxHeight * 0.5 * Math.cos(defaultPitch);
    const floor = -maxHeight * 0.002;
    const grid = [];
    for (let i = -5; i <= 5; i++) {
      const dx = x * i / 5;
      const dy = y * i / 5;
      grid.push(-x, dy, floor, -1, 1, x, dy, floor, -1, 1,
                dx, -y, floor, -1, 1, dx, y, floor, -1, 1);
    }
    meshCount = mesh.length / 5;
    gridCount = grid.length / 5;
    gl.bufferData(gl.ARRAY_BUFFER, (meshCount + gridCount) * 20, gl.STATIC_DRAW);
    gl.bufferSubData(gl.ARRAY_BUFFER, 0, mesh);
    gl.bufferSubData(gl.ARRAY_BUFFER, mesh.byteLength, Float32Array.from(grid));
    reset();
  }

  canvas.addEventListener("pointerdown", event => {
    if (event.button !== 0 || pointer) return;
    pointer = {id: event.pointerId, x: event.clientX, y: event.clientY};
    canvas.setPointerCapture(event.pointerId);
  });
  canvas.addEventListener("pointermove", event => {
    if (event.pointerId !== pointer?.id) return;
    rotate(event.clientX - pointer.x, event.clientY - pointer.y);
    pointer.x = event.clientX;
    pointer.y = event.clientY;
  });
  canvas.addEventListener("lostpointercapture", () => { pointer = undefined; });
  canvas.addEventListener("wheel", event => {
    event.preventDefault();
    zoomBy(event.deltaY < 0 ? 1.12 : 1 / 1.12);
  }, {passive: false});
  canvas.addEventListener("keydown", event => {
    const direction = {ArrowLeft: [-20, 0], ArrowRight: [20, 0], ArrowUp: [0, 15], ArrowDown: [0, -15]};
    if (direction[event.key]) {
      event.preventDefault();
      rotate(...direction[event.key]);
    }
  });
  new ResizeObserver(draw).observe(canvas);
  return {load, reset, zoomBy};
}
