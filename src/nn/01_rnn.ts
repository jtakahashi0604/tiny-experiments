export {};

type Tensor1D = number[];
type Tensor2D = number[][];

const Tensor = {
  random1D(rows: number): Tensor1D {
    const v: Tensor1D = [];

    for (let r = 0; r < rows; r++) {
      // Range: -1.0 to +1.0
      v.push(Math.random() * 2 - 1);
    }

    return v;
  },

  random2D(rows: number, cols: number): Tensor2D {
    const m: Tensor2D = [];

    for (let r = 0; r < rows; r++) {
      const row: Tensor1D = [];

      for (let c = 0; c < cols; c++) {
        // Range: -1.0 to +1.0
        row.push(Math.random() * 2 - 1);
      }

      m.push(row);
    }

    return m;
  },

  zeros1D(rows: number): Tensor1D {
    return new Array(rows).fill(0);
  },

  zeros2D(rows: number, cols: number): Tensor2D {
    return new Array(rows).fill(0).map(() => new Array(cols).fill(0));
  },

  mul_T1D_T2D(t1dA: Tensor1D, t2dA: Tensor2D): Tensor1D {
    const o: Tensor1D = [];

    for (let i = 0; i < t2dA.length; i++) {
      let sum = 0;

      for (let j = 0; j < t1dA.length; j++) {
        sum += t2dA[i][j] * t1dA[j];
      }

      o.push(sum);
    }

    return o;
  },

  add_T1D_T1D(t1dA: Tensor1D, t1dB: Tensor1D): Tensor1D {
    const o: Tensor1D = [];

    for (let i = 0; i < t1dA.length; i++) {
      o.push(t1dA[i] + t1dB[i]);
    }

    return o;
  },

  map1D(v: Tensor1D, fn: (value: number, index: number) => number): Tensor1D {
    const o: Tensor1D = [];

    for (let i = 0; i < v.length; i++) {
      o.push(fn(v[i], i));
    }

    return o;
  },

  join1D(t1dA: Tensor1D, t1dB: Tensor1D): Tensor1D {
    return [...t1dA, ...t1dB];
  },

  clip1D(v: Tensor1D, s: number, e: number): Tensor1D {
    return v.slice(s, e);
  },
};

interface Activation {
  fw(x: number): number;
  bw(y: number): number;
}

const Activations = {
  sigmoid: {
    fw(x: number): number {
      // NOTE: Why using exp?
      // ロジット
      // e
      return 1 / (1 + Math.exp(-x));
    },
    bw(y: number): number {
      return y * (1 - y);
    },
  } satisfies Activation,
};

class Layer {
  xSize: number;
  hSize: number;
  ySize: number;

  history_xs: Tensor1D[] = [];
  history_hs: Tensor1D[] = [];

  curr_h: Tensor1D;

  x_w: Tensor2D;
  h_w: Tensor2D;
  y_w: Tensor2D;
  h_b: Tensor1D;

  activation: Activation;

  constructor(
    xSize: number,
    hSize: number,
    ySize: number,
    activation: Activation,
  ) {
    this.xSize = xSize;
    this.hSize = hSize;
    this.ySize = ySize;
    this.activation = activation;

    this.x_w = Tensor.random2D(hSize, xSize);
    this.h_w = Tensor.random2D(hSize, hSize);
    this.y_w = Tensor.random2D(ySize, hSize);
    this.h_b = Tensor.random1D(hSize);

    this.curr_h = Tensor.zeros1D(hSize);
  }

  resetState() {
    this.history_xs = [];
    this.history_hs = [Tensor.zeros1D(this.hSize)];

    this.curr_h = Tensor.zeros1D(this.hSize);
  }

  fw(curr_x: Tensor1D): Tensor1D {
    const prev_h = this.curr_h;

    this.history_xs.push(curr_x);

    // step 1
    // h_{t-1} * h_w
    const h = Tensor.mul_T1D_T2D(prev_h, this.h_w);
    // step 2
    // x       * x_w
    const x = Tensor.mul_T1D_T2D(curr_x, this.x_w);

    // step 3
    const h_add_x = Tensor.add_T1D_T1D(h, x);

    // step 4
    const h_add_x_add_h_b = Tensor.add_T1D_T1D(h_add_x, this.h_b);

    // step 5
    const z = Tensor.map1D(h_add_x_add_h_b, (v) => this.activation.fw(v));

    const curr_h = [...z]; // Deep copy

    this.curr_h = curr_h;

    this.history_hs.push(curr_h);

    // step 6
    // y = curr_h * y_w
    const y = Tensor.mul_T1D_T2D(this.curr_h, this.y_w);

    return y;
  }

  bw(
    // 今の誤差 t
    curr_y_d: Tensor1D,
    // 次の誤差 t+1
    next_h_d: Tensor1D,
    r: number,
  ): Tensor1D {
    const curr_x = this.history_xs.pop()!;
    const curr_h = this.history_hs.pop()!;
    const prev_h = this.history_hs[this.history_hs.length - 1];

    // 連鎖律
    // step 0
    // curr_y_d
    const d1 = curr_y_d;

    // step 6
    // h * w
    // 微分: w
    // 残る: h
    for (let i = 0; i < this.ySize; i++) {
      for (let j = 0; j < this.hSize; j++) {
        const h = curr_h[j];

        const dw = d1[i] * h;

        this.y_w[i][j] -= r * dw;
      }
    }

    // h * w
    // 微分: h
    // 残る: w
    const d2 = Tensor.zeros1D(this.hSize);

    for (let j = 0; j < this.hSize; j++) {
      for (let i = 0; i < this.ySize; i++) {
        const w = this.y_w[i][j];

        const dh = d1[i] * w;

        d2[j] += dh;
      }

      d2[j] += next_h_d[j];
    }

    // step 5
    // アクティベーション
    // 微分
    const d3: Tensor1D = [];

    for (let i = 0; i < this.hSize; i++) {
      d3[i] = d2[i] * this.activation.bw(curr_h[i]);
    }

    // step 4
    for (let i = 0; i < this.hSize; i++) {
      this.h_b[i] -= r * d3[i] * 1;
    }

    // step 3 - 1
    const d4 = Tensor.zeros1D(this.hSize);

    for (let i = 0; i < this.hSize; i++) {
      for (let j = 0; j < this.xSize; j++) {
        // x * w
        // 微分: w
        // 残る: x
        this.x_w[i][j] -= r * d3[i] * curr_x[j];
      }

      for (let j = 0; j < this.hSize; j++) {
        d4[j] += d3[i] * this.h_w[i][j];

        // h * w
        // 微分: w
        // 残る: h
        this.h_w[i][j] -= r * d3[i] * prev_h[j];
      }
    }

    return d4;
  }
}

class Network {
  layer: Layer;

  constructor(xSize: number, hSize: number, ySize: number) {
    this.layer = new Layer(xSize, hSize, ySize, Activations.sigmoid);
  }

  fw(xSequence: Tensor1D[]): Tensor1D[] {
    this.layer.resetState();

    const ys: Tensor1D[] = [];

    for (const x of xSequence) {
      const y = this.layer.fw(x);
      ys.push(y);
    }

    return ys;
  }

  trainOne(xSequence: Tensor1D[], cSequence: Tensor1D[], r: number) {
    const ys = this.fw(xSequence);

    const ds: Tensor1D[] = [];

    for (let t = 0; t < ys.length; t++) {
      const d = Tensor.map1D(ys[t], (y, i) => {
        const c = cSequence[t][i];
        return y - c;
      });

      ds.push(d);
    }

    let next_h_d = Tensor.zeros1D(this.layer.hSize);
    let curr_y_d = Tensor.zeros1D(this.layer.hSize);

    // 時系列の後（next）から前（current）へと遷移
    for (let t = xSequence.length - 1; t >= 0; t--) {
      curr_y_d = ds[t];
      next_h_d = this.layer.bw(curr_y_d, next_h_d, r);
    }
  }

  trainAll(
    data: { x: Tensor1D[]; c: Tensor1D[] }[],
    epochs: number,
    r: number,
  ) {
    for (let i = 0; i < epochs; i++) {
      for (const sample of data) {
        this.trainOne(sample.x, sample.c, r);
      }
    }
  }

  predict(x: Tensor1D[]): Tensor1D[] {
    return this.fw(x);
  }
}

const trainingSamples = [
  { x: [[0.1], [0.4], [0.7]], c: [[0], [1], [1]] }, // Increase
  { x: [[0.3], [0.5], [0.9]], c: [[0], [1], [1]] }, // Increase
  { x: [[0.8], [0.5], [0.2]], c: [[0], [0], [0]] }, // Decrease
  { x: [[0.6], [0.3], [0.1]], c: [[0], [0], [0]] }, // Decrease
];

const network = new Network(1, 8, 1);

network.trainAll(trainingSamples, 10000, 0.1);

const samples = [
  [[0.2], [0.5], [0.8]], // Unknown Increase Data
  [[0.9], [0.6], [0.3]], // Unknown Decrease Data
];

for (const sample of samples) {
  const y = network.predict(sample);
  console.log(`x: ${sample}, y: ${y[y.length - 1][0].toFixed(4)}`);
}
