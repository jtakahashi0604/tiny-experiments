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

  convolve1D(t1dI: Tensor1D, t1dK: Tensor1D): number {
    let sum = 0;

    for (let i = 0; i < t1dI.length; i++) {
      sum += t1dI[i] * t1dK[i];
    }

    return sum;
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

class DenseLayer {
  w: Tensor2D;
  b: Tensor1D;
  activation: Activation;
  history_x: Tensor1D[] = [];
  history_y: Tensor1D[] = [];

  constructor(inSize: number, outSize: number, activation: Activation) {
    this.w = Tensor.random2D(outSize, inSize);
    this.b = Tensor.random1D(outSize);
    this.activation = activation;
  }

  fw(x: Tensor1D): Tensor1D {
    this.history_x.push(x);

    const y = Tensor.add_T1D_T1D(Tensor.mul_T1D_T2D(x, this.w), this.b);

    this.history_y.push(y);

    const a = Tensor.map1D(y, (v) => this.activation.fw(v));

    return a;
  }

  bw(d: Tensor1D, r: number): Tensor1D {
    const x = this.history_x.pop()!;
    const y = this.history_y.pop()!;

    const dy = Tensor.map1D(
      d,
      (g, i) => g * this.activation.bw(this.activation.fw(y[i])),
    );

    const dx = Tensor.zeros1D(x.length);

    for (let i = 0; i < this.w.length; i++) {
      for (let j = 0; j < x.length; j++) {
        const dw = dy[i] * x[j];

        dx[j] += dy[i] * this.w[i][j];

        this.w[i][j] -= r * dw;
      }

      this.b[i] -= r * dy[i];
    }

    return dx;
  }
}

class Layer {
  kSize: number;
  fSize: number;
  k: Tensor2D;
  b: Tensor1D;

  history_xs: Tensor1D[] = [];
  history_ys: Tensor1D[] = [];

  activation: Activation;

  constructor(kSize: number, fSize: number, activation: Activation) {
    this.kSize = kSize;
    this.fSize = fSize;
    this.k = Tensor.random2D(fSize, kSize);
    this.b = Tensor.random1D(fSize);
    this.activation = activation;
  }

  fw(input: Tensor1D): Tensor1D {
    this.history_xs.push(input);

    const oSize = input.length - this.kSize + 1;

    const a: Tensor1D = [];
    const y: Tensor1D = [];

    // each filter
    for (let f = 0; f < this.fSize; f++) {
      for (let i = 0; i < oSize; i++) {
        // window
        const w = input.slice(i, i + this.kSize);
        // convolve
        const v = Tensor.convolve1D(w, this.k[f]) + this.b[f];
        y.push(v);
        a.push(this.activation.fw(v));
      }
    }

    this.history_ys.push(y);

    return a;
  }

  bw(d: Tensor1D, r: number): Tensor1D {
    const x = this.history_xs.pop()!;
    const y = this.history_ys.pop()!;
    const ySize = x.length - this.kSize + 1;

    const dy = Tensor.map1D(
      d,
      (g, i) => g * this.activation.bw(this.activation.fw(y[i])),
    );

    const dx = Tensor.zeros1D(x.length);

    for (let f = 0; f < this.fSize; f++) {
      for (let i = 0; i < ySize; i++) {
        for (let k = 0; k < this.kSize; k++) {
          dx[i + k] += dy[f * ySize + i] * this.k[f][k];

          this.k[f][k] -= r * dy[f * ySize + i] * x[i + k];
        }

        this.b[f] -= r * dy[f * ySize + i];
      }
    }

    return dx;
  }
}

class Network {
  cLayer: Layer;
  dLayer: DenseLayer;

  constructor(xSize: number, kSize: number, fSize: number) {
    this.cLayer = new Layer(kSize, fSize, Activations.sigmoid);
    this.dLayer = new DenseLayer(
      fSize * (xSize - kSize + 1),
      1,
      Activations.sigmoid,
    );
  }

  fw(input: Tensor1D): Tensor1D {
    const y = this.cLayer.fw(input);
    return this.dLayer.fw(y);
  }

  trainOne(x: Tensor1D, c: Tensor1D, r: number) {
    const y = this.fw(x);

    const d1 = Tensor.map1D(y, (v, i) => v - c[i]);
    const d2 = this.dLayer.bw(d1, r);

    this.cLayer.bw(d2, r);
  }

  trainAll(data: { x: Tensor1D; c: Tensor1D }[], epochs: number, r: number) {
    for (let i = 0; i < epochs; i++) {
      for (const sample of data) {
        this.trainOne(sample.x, sample.c, r);
      }
    }
  }

  predict(input: Tensor1D): number {
    return this.fw(input)[0];
  }
}

const trainingSamples = [
  // has peak
  { x: [0.1, 0.9, 0.1, 0.2, 0.2], c: [1.0] },
  { x: [0.2, 0.1, 0.9, 0.1, 0.2], c: [1.0] },
  { x: [0.2, 0.2, 0.1, 0.9, 0.1], c: [1.0] },

  // has not peak
  { x: [0.1, 0.2, 0.3, 0.4, 0.5], c: [0.0] },
  { x: [0.9, 0.8, 0.7, 0.6, 0.5], c: [0.0] },
  { x: [0.5, 0.5, 0.5, 0.5, 0.5], c: [0.0] },
  { x: [0.9, 0.1, 0.9, 0.1, 0.9], c: [0.0] },
];

const network = new Network(5, 3, 2);

network.trainAll(trainingSamples, 10000, 0.1);

const samples = [
  [0.1, 0.2, 0.9, 0.2, 0.1],
  [0.5, 0.5, 0.5, 0.5, 0.5],
];

for (const sample of samples) {
  const y = network.predict(sample);
  console.log(`x: ${sample}, y: ${y.toFixed(4)}`);
}
