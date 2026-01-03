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

  mulT1DT2D(t1dA: Tensor1D, t2dA: Tensor2D): Tensor1D {
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

  add1DT1D(t1dA: Tensor1D, t1dB: Tensor1D): Tensor1D {
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

  history_xs: Tensor1D[] = [];
  history_hs: Tensor1D[] = [];

  curr_h: Tensor1D;

  x_w: Tensor2D;
  h_w: Tensor2D;
  h_b: Tensor1D;

  activation: Activation;

  constructor(xSize: number, hSize: number, activation: Activation) {
    this.xSize = xSize;
    this.hSize = hSize;
    this.activation = activation;

    this.x_w = Tensor.random2D(hSize, xSize);
    this.h_w = Tensor.random2D(hSize, hSize);
    this.h_b = Tensor.random1D(hSize);

    this.curr_h = Tensor.zeros1D(hSize);
  }

  resetState() {
    this.history_xs = [];
    this.history_hs = [];

    this.curr_h = Tensor.zeros1D(this.hSize);
  }

  fw(curr_x: Tensor1D): Tensor1D {
    const prev_h = this.curr_h;

    this.history_xs.push(curr_x);
    this.history_hs.push(prev_h);

    // 1. xW * x
    const x_w_mul_x = Tensor.mulT1DT2D(curr_x, this.x_w);
    // 2. hW * h_{t-1}
    const h_w_mul_h = Tensor.mulT1DT2D(prev_h, this.h_w);

    const z = Tensor.add1DT1D(Tensor.add1DT1D(x_w_mul_x, h_w_mul_h), this.h_b);

    this.curr_h = Tensor.map1D(z, (v) => this.activation.fw(v));

    return this.curr_h;
  }

  bw(
    // 今の誤差 t
    curr_h_d: Tensor1D,
    // 次の誤差 t+1
    next_h_d: Tensor1D,
    r: number,
  ): Tensor1D {
    const history_x = this.history_xs.pop()!;
    const history_h = this.history_hs.pop()!;

    const dz: Tensor1D = [];

    for (let yi = 0; yi < this.hSize; yi++) {
      dz[yi] =
        (curr_h_d[yi] + next_h_d[yi]) * this.activation.bw(this.curr_h[yi]);
    }

    const dx_total = Tensor.zeros1D(this.xSize);
    const dh_total = Tensor.zeros1D(this.hSize);

    for (let o = 0; o < this.hSize; o++) {
      for (let i = 0; i < this.xSize; i++) {
        // 合計を計算して、前のレイヤーにわたす
        dx_total[i] += dz[o] * this.x_w[o][i];
        // w の誤りを知りたい -> 微分すると x がのこる
        // x * w の w の偏微分
        const dw_i = dz[o] * history_x[i];
        // 更新
        this.x_w[o][i] -= r * dw_i;
      }

      for (let i = 0; i < this.hSize; i++) {
        // 合計を計算して、前のレイヤーにわたす
        dh_total[i] += dz[o] * this.h_w[o][i];
        // w の誤りを知りたい -> 微分すると h がのこる
        // h * w の w の偏微分
        const dw_i = dz[o] * history_h[i];
        // 更新
        this.h_w[o][i] -= r * dw_i;
      }

      // b の誤りを知りたい -> 微分すると 1 がのこる
      this.h_b[o] -= r * dz[o];
    }

    this.curr_h = history_h;

    return dh_total;
  }
}

class Network {
  layer: Layer;

  constructor(xSize: number, hSize: number) {
    this.layer = new Layer(xSize, hSize, Activations.sigmoid);
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
    let curr_h_d = Tensor.zeros1D(this.layer.hSize);

    // 時系列の後（next）から前（current）へと遷移
    for (let t = xSequence.length - 1; t >= 0; t--) {
      curr_h_d = ds[t];
      next_h_d = this.layer.bw(curr_h_d, next_h_d, r);
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

const network = new Network(1, 1);

network.trainAll(trainingSamples, 10000, 0.1);

const samples = [
  [[0.2], [0.5], [0.8]], // Unknown Increase Data
  [[0.9], [0.6], [0.3]], // Unknown Decrease Data
];

for (const sample of samples) {
  const y = network.predict(sample);
  console.log(`x: ${sample}, y: ${y[y.length - 1][0].toFixed(4)}`);
}
