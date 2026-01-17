const canvas = document.getElementById("draw");
const ctx = canvas.getContext("2d");
const mini = document.getElementById("mini");
const miniCtx = mini.getContext("2d");

const clearBtn = document.getElementById("clear");
const predictBtn = document.getElementById("predict");
const predEl = document.getElementById("pred");
const probsEl = document.getElementById("probs");
const hiddenEl = document.getElementById("hidden");

// init drawing canvas: black background
function resetCanvas() {
  ctx.fillStyle = "black";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
}
resetCanvas();

// drawing settings (MNIST-ish)
ctx.strokeStyle = "white";
ctx.lineWidth = 16;
ctx.lineCap = "round";
ctx.lineJoin = "round";

let drawing = false;
let lastX = 0, lastY = 0;

function getPos(e) {
  const rect = canvas.getBoundingClientRect();
  const x = (e.clientX - rect.left) * (canvas.width / rect.width);
  const y = (e.clientY - rect.top) * (canvas.height / rect.height);
  return { x, y };
}

canvas.addEventListener("pointerdown", (e) => {
  drawing = true;
  const p = getPos(e);
  lastX = p.x; lastY = p.y;
});

canvas.addEventListener("pointermove", (e) => {
  if (!drawing) return;
  const p = getPos(e);
  ctx.beginPath();
  ctx.moveTo(lastX, lastY);
  ctx.lineTo(p.x, p.y);
  ctx.stroke();
  lastX = p.x; lastY = p.y;
});

canvas.addEventListener("pointerup", () => drawing = false);
canvas.addEventListener("pointerleave", () => drawing = false);

clearBtn.addEventListener("click", () => {
  resetCanvas();
  predEl.textContent = "Predicted: —";
  probsEl.innerHTML = "";
  hiddenEl.innerHTML = "";
  miniCtx.clearRect(0, 0, mini.width, mini.height);
});

// Convert 280x280 canvas -> 28x28 grayscale -> 784 floats [0,1]
function getPixels784() {
  // Read full-res canvas pixels
  const full = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const data = full.data;

  // Convert to grayscale + find bounding box of "ink"
  // Since you draw white on black, grayscale near 1 = ink.
  const W = canvas.width, H = canvas.height;
  const threshold = 0.15; // ink threshold (0..1). tweak 0.10-0.25 if needed.

  let minX = W, minY = H, maxX = -1, maxY = -1;

  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const i = (y * W + x) * 4;
      const r = data[i], g = data[i + 1], b = data[i + 2];
      const gray = (r + g + b) / (3 * 255); // 0..1

      if (gray > threshold) {
        if (x < minX) minX = x;
        if (y < minY) minY = y;
        if (x > maxX) maxX = x;
        if (y > maxY) maxY = y;
      }
    }
  }

  // If nothing drawn, return all zeros
  if (maxX < minX || maxY < minY) {
    miniCtx.clearRect(0, 0, mini.width, mini.height);
    return new Array(784).fill(0);
  }

  // Add padding around the bounding box
  const pad = 28;
  minX = Math.max(0, minX - pad);
  minY = Math.max(0, minY - pad);
  maxX = Math.min(W - 1, maxX + pad);
  maxY = Math.min(H - 1, maxY + pad);

  const cropW = maxX - minX + 1;
  const cropH = maxY - minY + 1;

  // Put cropped region onto an offscreen canvas
  const cropCanvas = document.createElement("canvas");
  cropCanvas.width = cropW;
  cropCanvas.height = cropH;
  const cropCtx = cropCanvas.getContext("2d");

  cropCtx.drawImage(canvas, minX, minY, cropW, cropH, 0, 0, cropW, cropH);

  // Resize the cropped digit to fit into 20x20 while preserving aspect ratio
  const target = 22;
  const scaledCanvas = document.createElement("canvas");
  scaledCanvas.width = target;
  scaledCanvas.height = target;
  const scaledCtx = scaledCanvas.getContext("2d");

  // black background
  scaledCtx.fillStyle = "black";
  scaledCtx.fillRect(0, 0, target, target);

  const scale = Math.min(target / cropW, target / cropH);
  const newW = Math.max(1, Math.round(cropW * scale));
  const newH = Math.max(1, Math.round(cropH * scale));
  const dx = Math.floor((target - newW) / 2);
  const dy = Math.floor((target - newH) / 2);

  scaledCtx.drawImage(cropCanvas, 0, 0, cropW, cropH, dx, dy, newW, newH);

  // Center the 20x20 into a 28x28 canvas
  const outCanvas = document.createElement("canvas");
  outCanvas.width = 28;
  outCanvas.height = 28;
  const outCtx = outCanvas.getContext("2d");

  outCtx.fillStyle = "black";
  outCtx.fillRect(0, 0, 28, 28);

  outCtx.drawImage(scaledCanvas, 4, 4); // 28-20=8, so offset by 4 to center

  // Show what the model sees (scaled up)
  miniCtx.imageSmoothingEnabled = false;
  miniCtx.clearRect(0, 0, mini.width, mini.height);
  miniCtx.drawImage(outCanvas, 0, 0, mini.width, mini.height);

  // Convert 28x28 to 784 floats
  const img = outCtx.getImageData(0, 0, 28, 28);
  const out = img.data;

  const pixels = new Array(784);
  for (let i = 0; i < 784; i++) {
    const r = out[i * 4 + 0];
    const g = out[i * 4 + 1];
    const b = out[i * 4 + 2];
    const gray = (r + g + b) / (3 * 255); // 0..1
    pixels[i] = gray;
  }

  return pixels;
}


function renderProbs(probs) {
  probsEl.innerHTML = "";
  for (let i = 0; i < 10; i++) {
    const row = document.createElement("div");
    row.className = "bar";

    const label = document.createElement("div");
    label.textContent = String(i);

    const track = document.createElement("div");
    track.className = "track";

    const fill = document.createElement("div");
    fill.className = "fill";
    fill.style.width = (probs[i] * 100).toFixed(1) + "%";
    track.appendChild(fill);

    const val = document.createElement("div");
    val.className = "small";
    val.textContent = (probs[i] * 100).toFixed(1) + "%";

    row.appendChild(label);
    row.appendChild(track);
    row.appendChild(val);
    probsEl.appendChild(row);
  }
}

function renderHidden(hidden) {
  hiddenEl.innerHTML = "";
  // normalize for display (not affecting model)
  let maxVal = 0;
  for (const v of hidden) maxVal = Math.max(maxVal, v);
  maxVal = Math.max(maxVal, 1e-6);

  for (let i = 0; i < hidden.length; i++) {
    const c = document.createElement("div");
    c.className = "cell";
    const intensity = hidden[i] / maxVal; // 0..1
    c.style.opacity = (0.15 + 0.85 * intensity).toFixed(3);
    hiddenEl.appendChild(c);
  }
}

predictBtn.addEventListener("click", async () => {
  const pixels = getPixels784();

  const res = await fetch("/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ pixels })
  });

  const out = await res.json();
  if (!res.ok) {
    predEl.textContent = "Error: " + (out.error || "unknown");
    return;
  }

  predEl.textContent = "Predicted: " + out.pred;
  renderProbs(out.probs);
  renderHidden(out.hidden);
});
