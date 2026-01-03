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
  ySize: number;

  x: Tensor1D;
  y: Tensor1D;

  w: Tensor2D;
  b: Tensor1D;

  activation: Activation;

  constructor(xSize: number, ySize: number, activation: Activation) {
    this.xSize = xSize;
    this.ySize = ySize;
    this.activation = activation;

    this.w = Tensor.random2D(ySize, xSize);
    this.b = Tensor.random1D(ySize);
  }

  fw(x: Tensor1D): Tensor1D {
    const z = Tensor.add1DT1D(Tensor.mulT1DT2D(x, this.w), this.b);
    // const y = z;
    const y = Tensor.map1D(z, (v) => this.activation.fw(v));

    this.x = x;
    this.y = y;

    return this.y;
  }

  bw(d: Tensor1D, r: number): Tensor1D {
    const dz: Tensor1D = [];

    // 微分計算 - 活性化関数
    // dz = d;
    for (let yi = 0; yi < this.ySize; yi++) {
      dz[yi] = d[yi] * this.activation.bw(this.y[yi]);
    }

    // 微分計算 - x, w, b
    const dx_total = Tensor.zeros1D(this.xSize);

    for (let yi = 0; yi < this.ySize; yi++) {
      for (let xi = 0; xi < this.xSize; xi++) {
        // x の誤りを知りたい -> 微分すると w がのこる
        const dx_i = dz[yi] * this.w[yi][xi];
        // 合計を計算して、前のレイヤーにわたす
        dx_total[xi] += dx_i;

        // w の誤りを知りたい -> 微分すると x がのこる
        const dw_i = dz[yi] * this.x[xi];
        // NOTE: Why using minus?
        // 勾配がのぼっている（+）なら、重みを減らす
        // 勾配が下がっている（-）なら、重みを増やす
        this.w[yi][xi] -= r * dw_i;
      }

      // b の誤りを知りたい -> 微分すると 1 がのこる
      this.b[yi] -= r * dz[yi];
    }

    // 微分計算 - x を前のレイヤーにわたす
    return dx_total;
  }
}

class Network {
  layer1: Layer;
  layer2: Layer;

  constructor(xSize: number, hSize: number, ySize: number) {
    this.layer1 = new Layer(xSize, hSize, Activations.sigmoid);
    this.layer2 = new Layer(hSize, ySize, Activations.sigmoid);
  }

  fw(x1: Tensor1D): Tensor1D {
    const y1 = this.layer1.fw(x1);
    const y2 = this.layer2.fw(y1);
    return y2;
  }

  trainOne(x: Tensor1D, c: Tensor1D, r: number, epoch: number) {
    const y = this.fw(x);

    const dl: Tensor1D = [];

    let sumError = 0;

    for (let i = 0; i < y.length; i++) {
      // 微分計算 - 損失関数
      // 原始関数: 1/2 * (y - c)^2
      // 微分関数: y - c
      dl[i] = y[i] - c[i];

      sumError += dl[i] ** 2;
    }

    const d2 = this.layer2.bw(dl, r);
    const d1 = this.layer1.bw(d2, r);
  }

  trainAll(data: { x: Tensor1D; c: Tensor1D }[], epochs: number, r: number) {
    for (let i = 0; i < epochs; i++) {
      for (const sample of data) {
        this.trainOne(sample.x, sample.c, r, i);
      }
    }
  }

  predict(x: Tensor1D): Tensor1D {
    return this.fw(x);
  }
}

const trainingSamples = [
  { x: [0, 0], c: [0] },
  { x: [0, 1], c: [1] },
  { x: [1, 0], c: [1] },
  { x: [1, 1], c: [0] },
];

const network = new Network(2, 2, 1);

network.trainAll(trainingSamples, 10000, 0.5);

const samples = [
  [0, 0],
  [0, 1],
  [1, 0],
  [1, 1],
];

for (const sample of samples) {
  const y = network.predict(sample);

  console.log(`x: ${sample}, y: ${y[0].toFixed(4)}`);
}
