import { useMemo, useState } from "react";

// Single-file React demo: Perceptron epoch visualizer
// - Generates 2D data
// - Trains a perceptron while saving a snapshot each epoch
// - Lets you scrub through epochs with a slider

function randn() {
  let u = 0,
    v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

function makeGaussianData({
  nPerClass = 50,
  centerPos = [2, 2],
  centerNeg = [-2, -2],
  spread = 1.0,
}) {
  const X = [];
  const y = [];

  for (let i = 0; i < nPerClass; i++) {
    X.push([centerPos[0] + randn() * spread, centerPos[1] + randn() * spread]);
    y.push(1);
  }
  for (let i = 0; i < nPerClass; i++) {
    X.push([centerNeg[0] + randn() * spread, centerNeg[1] + randn() * spread]);
    y.push(-1);
  }

  return { X, y };
}

function shuffleTogether(X, y) {
  const idx = [...Array(X.length).keys()];
  for (let i = idx.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [idx[i], idx[j]] = [idx[j], idx[i]];
  }
  return {
    X: idx.map((i) => X[i]),
    y: idx.map((i) => y[i]),
  };
}

function predictPoint(w, b, x) {
  const z = w[0] * x[0] + w[1] * x[1] + b;
  return z >= 0 ? 1 : -1;
}

function accuracy(X, y, w, b) {
  let correct = 0;
  for (let i = 0; i < X.length; i++) {
    if (predictPoint(w, b, X[i]) === y[i]) correct++;
  }
  return correct / X.length;
}

function trainPerceptronHistory(
  X0,
  y0,
  { maxEpochs = 50, learningRate = 1, shuffleEachEpoch = true } = {},
) {
  let { X, y } = { X: X0.slice(), y: y0.slice() };
  let w = [0, 0];
  let b = 0;
  const history = [];

  // Save initial state
  history.push({
    epoch: 0,
    w: [...w],
    b,
    errors: null,
    accuracy: accuracy(X, y, w, b),
    note: "Initial state",
  });

  for (let epoch = 1; epoch <= maxEpochs; epoch++) {
    if (shuffleEachEpoch) {
      const shuffled = shuffleTogether(X, y);
      X = shuffled.X;
      y = shuffled.y;
    }

    let errors = 0;

    for (let i = 0; i < X.length; i++) {
      const x = X[i];
      const yTrue = y[i];
      const yPred = predictPoint(w, b, x);

      if (yPred !== yTrue) {
        w = [
          w[0] + learningRate * yTrue * x[0],
          w[1] + learningRate * yTrue * x[1],
        ];
        b = b + learningRate * yTrue;
        errors++;
      }
    }

    history.push({
      epoch,
      w: [...w],
      b,
      errors,
      accuracy: accuracy(X, y, w, b),
      note: errors === 0 ? "Converged" : "Updated after one full pass",
    });

    if (errors === 0) break;
  }

  return { history, X, y };
}

function getBounds(X) {
  const xs = X.map((p) => p[0]);
  const ys = X.map((p) => p[1]);
  const minX = Math.min(...xs) - 1;
  const maxX = Math.max(...xs) + 1;
  const minY = Math.min(...ys) - 1;
  const maxY = Math.max(...ys) + 1;
  return { minX, maxX, minY, maxY };
}

function mapToSvg(x, y, bounds, width, height, pad = 30) {
  const sx =
    pad + ((x - bounds.minX) / (bounds.maxX - bounds.minX)) * (width - pad * 2);
  const sy =
    height -
    pad -
    ((y - bounds.minY) / (bounds.maxY - bounds.minY)) * (height - pad * 2);
  return [sx, sy];
}

function boundarySegment(w, b, bounds) {
  const pts = [];
  const { minX, maxX, minY, maxY } = bounds;

  if (Math.abs(w[1]) > 1e-9) {
    const yAtMinX = -(w[0] * minX + b) / w[1];
    const yAtMaxX = -(w[0] * maxX + b) / w[1];
    if (yAtMinX >= minY && yAtMinX <= maxY) pts.push([minX, yAtMinX]);
    if (yAtMaxX >= minY && yAtMaxX <= maxY) pts.push([maxX, yAtMaxX]);
  }
  if (Math.abs(w[0]) > 1e-9) {
    const xAtMinY = -(w[1] * minY + b) / w[0];
    const xAtMaxY = -(w[1] * maxY + b) / w[0];
    if (xAtMinY >= minX && xAtMinY <= maxX) pts.push([xAtMinY, minY]);
    if (xAtMaxY >= minX && xAtMaxY <= maxX) pts.push([xAtMaxY, maxY]);
  }

  const unique = [];
  for (const p of pts) {
    if (
      !unique.some(
        (q) => Math.abs(q[0] - p[0]) < 1e-6 && Math.abs(q[1] - p[1]) < 1e-6,
      )
    ) {
      unique.push(p);
    }
  }

  return unique.length >= 2 ? [unique[0], unique[1]] : null;
}

function Plot({ X, y, w, b }) {
  const width = 720;
  const height = 520;
  const bounds = getBounds(X);
  const segment = boundarySegment(w, b, bounds);

  const xAxis =
    bounds.minY <= 0 && bounds.maxY >= 0
      ? [bounds.minX, 0, bounds.maxX, 0]
      : null;
  const yAxis =
    bounds.minX <= 0 && bounds.maxX >= 0
      ? [0, bounds.minY, 0, bounds.maxY]
      : null;

  return (
    <svg
      viewBox={`0 0 ${width} ${height}`}
      className="w-full h-auto rounded-2xl border bg-white shadow-sm"
    >
      {xAxis &&
        (() => {
          const [x1, y1] = mapToSvg(xAxis[0], xAxis[1], bounds, width, height);
          const [x2, y2] = mapToSvg(xAxis[2], xAxis[3], bounds, width, height);
          return (
            <line
              x1={x1}
              y1={y1}
              x2={x2}
              y2={y2}
              stroke="#cbd5e1"
              strokeWidth="1"
            />
          );
        })()}
      {yAxis &&
        (() => {
          const [x1, y1] = mapToSvg(yAxis[0], yAxis[1], bounds, width, height);
          const [x2, y2] = mapToSvg(yAxis[2], yAxis[3], bounds, width, height);
          return (
            <line
              x1={x1}
              y1={y1}
              x2={x2}
              y2={y2}
              stroke="#cbd5e1"
              strokeWidth="1"
            />
          );
        })()}

      {segment &&
        (() => {
          const [p1, p2] = segment;
          const [x1, y1] = mapToSvg(p1[0], p1[1], bounds, width, height);
          const [x2, y2] = mapToSvg(p2[0], p2[1], bounds, width, height);
          return (
            <line
              x1={x1}
              y1={y1}
              x2={x2}
              y2={y2}
              stroke="#2563eb"
              strokeWidth="3"
            />
          );
        })()}

      {X.map((p, i) => {
        const [cx, cy] = mapToSvg(p[0], p[1], bounds, width, height);
        const fill = y[i] === 1 ? "#2563eb" : "#f97316";
        return (
          <circle key={i} cx={cx} cy={cy} r="5" fill={fill} opacity="0.9" />
        );
      })}
    </svg>
  );
}

export default function PerceptronEpochVisualizer() {
  const [nPerClass, setNPerClass] = useState(50);
  const [spread, setSpread] = useState(1.0);
  const [centerGap, setCenterGap] = useState(2.0);
  const [maxEpochs, setMaxEpochs] = useState(40);
  const [shuffleEachEpoch, setShuffleEachEpoch] = useState(true);
  const [seedTick, setSeedTick] = useState(0);
  const [epochIndex, setEpochIndex] = useState(0);

  const result = useMemo(() => {
    const centerPos = [centerGap, centerGap];
    const centerNeg = [-centerGap, -centerGap];
    const { X, y } = makeGaussianData({
      nPerClass,
      centerPos,
      centerNeg,
      spread,
    });
    const trained = trainPerceptronHistory(X, y, {
      maxEpochs,
      learningRate: 1,
      shuffleEachEpoch,
    });
    return trained;
  }, [nPerClass, spread, centerGap, maxEpochs, shuffleEachEpoch, seedTick]);

  const maxIdx = result.history.length - 1;
  const clampedEpoch = Math.min(epochIndex, maxIdx);
  const state = result.history[clampedEpoch];

  return (
    <div className="min-h-screen bg-slate-50 p-6">
      <div className="mx-auto max-w-6xl space-y-6">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">
            Perceptron Epoch Visualizer
          </h1>
        </div>

        <div className="flex items-center flex-row">
          <div className="rounded-2xl border bg-white p-4 shadow-sm space-y-5">
            <div>
              <label className="text-sm font-medium">
                Points per class: {nPerClass}
              </label>
              <input
                type="range"
                min="10"
                max="120"
                value={nPerClass}
                onChange={(e) => setNPerClass(Number(e.target.value))}
                className="mt-2 w-full"
              />
            </div>

            <div>
              <label className="text-sm font-medium">
                Spread: {spread.toFixed(1)}
              </label>
              <input
                type="range"
                min="0.3"
                max="2.5"
                step="0.1"
                value={spread}
                onChange={(e) => setSpread(Number(e.target.value))}
                className="mt-2 w-full"
              />
            </div>

            <div>
              <label className="text-sm font-medium">
                Center gap: {centerGap.toFixed(1)}
              </label>
              <input
                type="range"
                min="0.5"
                max="4"
                step="0.1"
                value={centerGap}
                onChange={(e) => setCenterGap(Number(e.target.value))}
                className="mt-2 w-full"
              />
            </div>

            <div>
              <label className="text-sm font-medium">
                Max epochs: {maxEpochs}
              </label>
              <input
                type="range"
                min="5"
                max="100"
                value={maxEpochs}
                onChange={(e) => setMaxEpochs(Number(e.target.value))}
                className="mt-2 w-full"
              />
            </div>

            <label className="flex items-center gap-3 text-sm">
              <input
                type="checkbox"
                checked={shuffleEachEpoch}
                onChange={(e) => setShuffleEachEpoch(e.target.checked)}
              />
              Shuffle each epoch
            </label>

            <div className="flex gap-3">
              <button
                onClick={() => {
                  setSeedTick((v) => v + 1);
                  setEpochIndex(0);
                }}
                className="rounded-xl bg-slate-900 px-4 py-2 text-white shadow-sm hover:bg-slate-800"
              >
                Regenerate data
              </button>
              <button
                onClick={() => setEpochIndex(0)}
                className="rounded-xl border px-4 py-2 hover:bg-slate-50"
              >
                Reset to epoch 0
              </button>
            </div>

            <div className="rounded-xl bg-slate-50 p-4 text-sm space-y-1">
              <div>
                <span className="font-medium">Current epoch:</span>{" "}
                {state.epoch}
              </div>
              <div>
                <span className="font-medium">Errors this epoch:</span>{" "}
                {state.errors === null ? "-" : state.errors}
              </div>
              <div>
                <span className="font-medium">Accuracy:</span>{" "}
                {(state.accuracy * 100).toFixed(1)}%
              </div>
              <div>
                <span className="font-medium">w:</span> [{state.w[0].toFixed(3)}
                , {state.w[1].toFixed(3)}]
              </div>
              <div>
                <span className="font-medium">b:</span> {state.b.toFixed(3)}
              </div>
              <div>
                <span className="font-medium">Status:</span> {state.note}
              </div>
            </div>
          </div>

          <div className="space-y-4">
            <div className="rounded-2xl border bg-white p-4 shadow-sm">
              <Plot X={result.X} y={result.y} w={state.w} b={state.b} />
            </div>

            <div className="rounded-2xl border bg-white p-4 shadow-sm space-y-3">
              <div className="flex items-center justify-between text-sm">
                <span className="font-medium">Epoch slider</span>
                <span>
                  {clampedEpoch} / {maxIdx}
                </span>
              </div>
              <input
                type="range"
                min="0"
                max={maxIdx}
                value={clampedEpoch}
                onChange={(e) => setEpochIndex(Number(e.target.value))}
                className="w-full"
              />
              <div className="text-sm text-slate-600">
                The training loop stops early when an epoch has 0 errors. If the
                data is not linearly separable, it runs until max epochs.
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
