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
    const v = new Array(rows).fill(0);
    return v;
  },

  zeros2D(rows: number, cols: number): Tensor2D {
    const m = new Array(rows).fill(0).map(() => new Array(cols).fill(0));
    return m;
  },

  identity1D: (size: number): Tensor1D => {
    const v = new Array(size).fill(0);

    for (let i = 0; i < size; i++) {
      v[i] = 1;
    }

    return v;
  },

  identity2D: (size: number): Tensor2D => {
    const m = new Array(size).fill(0).map(() => new Array(size).fill(0));

    for (let i = 0; i < size; i++) {
      m[i][i] = 1;
    }

    return m;
  },

  transpose2D: (t2d: Tensor2D): Tensor2D => {
    const m = new Array(t2d[0].length)
      .fill(0)
      .map(() => new Array(t2d.length).fill(0));

    for (let i = 0; i < t2d.length; i++) {
      for (let j = 0; j < t2d[0].length; j++) {
        m[j][i] = t2d[i][j];
      }
    }

    return m;
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

  mul_T2D_T2D(t2dA: Tensor2D, t2dB: Tensor2D): Tensor2D {
    const o = this.zeros2D(t2dA.length, t2dB[0].length);

    for (let i = 0; i < t2dA.length; i++) {
      for (let k = 0; k < t2dB.length; k++) {
        for (let j = 0; j < t2dB[0].length; j++)
          o[i][j] += t2dA[i][k] * t2dB[k][j];
      }
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

type SSMStep = { A_step: Tensor2D; B_step: Tensor1D };

const Scanner = {
  combine: (s1: SSMStep, s2: SSMStep): SSMStep => {
    // h1 = h_{0} * step1 A + step1 B
    // h2 = h_{1}               * step2 A + step2 B
    // h2 = (step1 A + step1 B) * step2 A + step2 B
    // h2 = h_{0} * (step1 A * step2 A) + (step1 B * step2 A) + step2 B
    // A_step is    (step1 A * step2 A)
    // B_step is                          (step1 B * step2 A) + step2 B

    return {
      A_step: Tensor.mul_T2D_T2D(s2.A_step, s1.A_step),
      B_step: Tensor.add_T1D_T1D(
        Tensor.mul_T1D_T2D(s1.B_step, s2.A_step),
        s2.B_step,
      ),
    };
  },

  parallelPrefixScan: (steps: SSMStep[]): SSMStep[] => {
    const combinedSteps: SSMStep[] = [...steps];

    for (let offset = 1; offset < steps.length; offset *= 2) {
      for (let i = steps.length - 1; i >= offset; i--) {
        combinedSteps[i] = Scanner.combine(
          combinedSteps[i - offset],
          combinedSteps[i],
        );
      }
    }

    return combinedSteps;
  },
};

class Layer {
  xSize: number;
  hSize: number;
  ySize: number;

  // history_xs: Tensor1D[] = [];
  // history_hs: Tensor1D[] = [];
  // history_ys: Tensor1D[] = [];
  // history_zs: Tensor1D[] = [];
  // history_ds: Tensor1D[] = [];

  // curr_h: Tensor1D;

  A: Tensor2D; // h_w
  B: Tensor2D; // x_w
  C: Tensor2D; // y_w
  h_b: Tensor1D;

  delta_w: Tensor2D;
  delta_b: Tensor1D;

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

    this.A = Tensor.random2D(hSize, hSize);
    this.B = Tensor.random2D(hSize, xSize);
    this.C = Tensor.random2D(ySize, hSize);
    this.h_b = Tensor.random1D(hSize);

    this.delta_w = Tensor.random2D(hSize, xSize);
    this.delta_b = Tensor.random1D(hSize);

    // this.curr_h = Tensor.zeros1D(hSize);
  }

  getStep(x: Tensor1D): SSMStep {
    const d_logits = Tensor.add_T1D_T1D(
      Tensor.mul_T1D_T2D(x, this.delta_w),
      this.delta_b,
    );

    const deltas = d_logits.map((v) => this.activation.fw(v));

    // 忘却ゲート
    // 次の時刻に今の時刻をどれぐらい忘却するか？
    // A_step = I + A * deltas
    const A_init = Tensor.identity2D(this.hSize);
    const A_step = A_init;

    for (let i = 0; i < this.hSize; i++) {
      for (let j = 0; j < this.hSize; j++) {
        // biome-ignore format: align
        A_step[i][j] += this.A[i][j] * deltas[i];
      }
    }

    // 入力ゲート
    // 次の時刻に今の時刻をどれぐらい入力するか？
    // B_step = I + B * deltas
    const B_init = [...this.h_b];
    const B_step = B_init;

    for (let i = 0; i < this.hSize; i++) {
      for (let j = 0; j < this.xSize; j++) {
        // biome-ignore format: align
        B_step[i]    += this.B[i][j] * deltas[i] * x[j];
      }
    }

    return { A_step, B_step };
  }

  applyGradients(
    xSeq: Tensor1D[],
    hSeq: Tensor1D[],
    dySeq: Tensor1D[],
    dhSeq: Tensor1D[],
    r: number,
  ) {
    let prev_h = Tensor.zeros1D(this.hSize);

    for (let t = 0; t < xSeq.length; t++) {
      const curr_x = xSeq[t];
      const curr_h = hSeq[t];
      const dh = dhSeq[t];
      const dy = dySeq[t];

      const d_logits = Tensor.add_T1D_T1D(
        Tensor.mul_T1D_T2D(curr_x, this.delta_w),
        this.delta_b,
      );

      const deltas = d_logits.map((v) => this.activation.fw(v));

      const z = Tensor.add_T1D_T1D(
        Tensor.mul_T1D_T2D(prev_h, this.A),
        Tensor.mul_T1D_T2D(curr_x, this.B),
      );

      const d_delta = Tensor.map1D(
        dh,
        (g, i) => g * z[i] * this.activation.bw(deltas[i]),
      );

      for (let i = 0; i < this.hSize; i++) {
        this.delta_b[i] -= r * d_delta[i];

        for (let j = 0; j < this.xSize; j++) {
          this.delta_w[i][j] -= r * d_delta[i] * curr_x[j];
        }
      }

      const g_base = Tensor.map1D(dh, (g, i) => g * deltas[i]);

      for (let i = 0; i < this.hSize; i++) {
        this.h_b[i] -= r * dh[i];

        this.C[0][i] -= r * dy[0] * curr_h[i];

        for (let j = 0; j < this.hSize; j++) {
          this.A[i][j] -= r * g_base[i] * prev_h[j];
        }

        for (let j = 0; j < this.xSize; j++) {
          this.B[i][j] -= r * g_base[i] * curr_x[j];
        }
      }

      prev_h = curr_h;
    }
  }

  // resetState() {
  //   this.history_xs = [];
  //   this.history_hs = [Tensor.zeros1D(this.hSize)];
  //   this.history_ys = [];
  //   this.history_zs = [];
  //   this.history_ds = [];

  //   this.curr_h = Tensor.zeros1D(this.hSize);
  // }

  // fw(curr_x: Tensor1D): Tensor1D {
  //   const prev_h = this.curr_h;

  //   this.history_xs.push(curr_x);

  //   // 変化量の計算
  //   const h = Tensor.mul_T1D_T2D(prev_h, this.A);
  //   // 変化量の計算
  //   const x = Tensor.mul_T1D_T2D(curr_x, this.B);
  //   // 変化量の計算 - 合計
  //   const z = Tensor.add_T1D_T1D(h, x);

  //   this.history_zs.push(z);

  //   // 変化量を変化率として扱う
  //   // 差分量 = 変化率 * スコープ
  //   const delta_logits = Tensor.add_T1D_T1D(
  //     Tensor.mul_T1D_T2D(curr_x, this.delta_w),
  //     this.delta_b,
  //   );

  //   // ゲート
  //   const deltas = delta_logits.map((v) => this.activation.fw(v));

  //   this.history_ds.push(deltas);

  //   const d = z.map((v, i) => v * deltas[i]);

  //   // 現在の場所 + 差分量 + バイアス
  //   const curr_h = [
  //     ...Tensor.add_T1D_T1D(Tensor.add_T1D_T1D(prev_h, d), this.h_b),
  //   ]; // Deep copy

  //   this.curr_h = curr_h;

  //   this.history_hs.push(curr_h);

  //   // y = curr_h * y_w
  //   const y = Tensor.mul_T1D_T2D(this.curr_h, this.C);

  //   const curr_y = y;

  //   this.history_ys.push(curr_y);

  //   const a = Tensor.map1D(y, (v) => this.activation.fw(v));

  //   return a;
  // }

  // bw(
  //   // 今の誤差 t
  //   curr_y_d: Tensor1D,
  //   // 次の誤差 t+1
  //   next_h_d: Tensor1D,
  //   r: number,
  // ): Tensor1D {
  //   const curr_x = this.history_xs.pop()!;
  //   const curr_h = this.history_hs.pop()!;
  //   const curr_y = this.history_ys.pop()!;
  //   const curr_z = this.history_zs.pop()!;
  //   const curr_d = this.history_ds.pop()!;
  //   const prev_h = this.history_hs[this.history_hs.length - 1];

  //   const d1 = Tensor.map1D(curr_y_d, (val, i) => {
  //     return val * this.activation.bw(this.activation.fw(curr_y[i]));
  //   });

  //   // h * C
  //   // 微分: C
  //   // 残る: h
  //   for (let i = 0; i < this.ySize; i++) {
  //     for (let j = 0; j < this.hSize; j++) {
  //       this.C[i][j] -= r * d1[i] * curr_h[j];
  //     }
  //   }

  //   // h * C
  //   // 微分: h
  //   // 残る: C
  //   const d2 = Tensor.zeros1D(this.hSize);

  //   for (let j = 0; j < this.hSize; j++) {
  //     for (let i = 0; i < this.ySize; i++) {
  //       d2[j] += d1[i] * this.C[i][j];
  //     }
  //     d2[j] += next_h_d[j];
  //   }

  //   const d3 = [...d2];

  //   // スコープ
  //   for (let i = 0; i < this.hSize; i++) {
  //     const d_delta = d3[i] * curr_z[i] * this.activation.bw(curr_d[i]);

  //     this.delta_b[i] -= r * d_delta;

  //     for (let j = 0; j < this.xSize; j++) {
  //       this.delta_w[i][j] -= r * d_delta * curr_x[j];
  //     }
  //   }

  //   // バイアス
  //   for (let i = 0; i < this.hSize; i++) {
  //     this.h_b[i] -= r * d3[i] * 1;
  //   }

  //   const d4 = [...d3];

  //   for (let i = 0; i < this.hSize; i++) {
  //     const d_base = d3[i] * curr_d[i];

  //     for (let j = 0; j < this.xSize; j++) {
  //       // x * B
  //       // 微分: B
  //       // 残る: x
  //       this.B[i][j] -= r * d_base * curr_x[j];
  //     }
  //     for (let j = 0; j < this.hSize; j++) {
  //       d4[j] += d_base * this.A[i][j];
  //       // h * A
  //       // 微分: A
  //       // 残る: h
  //       this.A[i][j] -= r * d_base * prev_h[j];
  //     }
  //   }

  //   return d4;
  // }
}

class Network {
  layer: Layer;

  delta_w: Tensor2D;
  delta_b: Tensor1D;

  constructor(xSize: number, hSize: number, ySize: number) {
    this.layer = new Layer(xSize, hSize, ySize, Activations.sigmoid);

    this.delta_w = Tensor.random2D(hSize, xSize);
    this.delta_b = Tensor.random1D(hSize);
  }

  trainOne(xSeq: Tensor1D[], cSeq: Tensor1D[], r: number) {
    const initialHiddenState = Tensor.zeros1D(this.layer.hSize);

    const steps: SSMStep[] = xSeq.map((x) => this.layer.getStep(x));

    const prefixes = Scanner.parallelPrefixScan(steps);

    const hSeq = prefixes.map((s) => {
      const h = Tensor.mul_T1D_T2D(initialHiddenState, s.A_step);

      return Tensor.add_T1D_T1D(h, s.B_step);
    });

    const ySeq = hSeq.map((h) => {
      const y = Tensor.mul_T1D_T2D(h, this.layer.C);
      return y;
    });

    const aSeq = ySeq.map((y) => {
      const a = Tensor.map1D(y, (v) => this.layer.activation.fw(v));
      return a;
    });

    const dySeq = aSeq.map((a, t) => {
      return Tensor.map1D(a, (v, i) => {
        const error = v - cSeq[t][i];
        return error * this.layer.activation.bw(v);
      });
    });

    const dhSeq_from_dySeq = dySeq.map((d) =>
      Tensor.mul_T1D_T2D(d, Tensor.transpose2D(this.layer.C)),
    );

    const dhSeq = new Array(xSeq.length);

    let prev_y_d = Tensor.zeros1D(this.layer.hSize);

    for (let t = xSeq.length - 1; t >= 0; t--) {
      // 全誤差 = (今の誤差) + (次の誤差)
      // 次の誤差には A_step をかける（忘却ゲート）
      const next_A_step = steps[t + 1] ? steps[t + 1].A_step : null;

      if (next_A_step) {
        prev_y_d = Tensor.add_T1D_T1D(
          dhSeq_from_dySeq[t],
          Tensor.mul_T1D_T2D(prev_y_d, next_A_step),
        );
      } else {
        // 最後の時刻
        prev_y_d = dhSeq_from_dySeq[t];
      }

      dhSeq[t] = prev_y_d;
    }

    this.layer.applyGradients(xSeq, hSeq, dySeq, dhSeq, r);
  }

  predict(xSeq: Tensor1D[]): Tensor1D[] {
    const steps: SSMStep[] = [];

    for (const x of xSeq) {
      const step = this.layer.getStep(x);
      steps.push(step);
    }

    const prefixes = Scanner.parallelPrefixScan(steps);

    const result: Tensor1D[] = [];

    const g1 = Tensor.zeros1D(this.layer.hSize);

    for (const s of prefixes) {
      const g2 = Tensor.mul_T1D_T2D(g1, s.A_step);
      const g3 = Tensor.add_T1D_T1D(g2, s.B_step);

      const y = Tensor.mul_T1D_T2D(g3, this.layer.C);

      const a = Tensor.map1D(y, (v) => this.layer.activation.fw(v));

      result.push(a);
    }

    return result;
  }

  // fw(xSequence: Tensor1D[]): Tensor1D[] {
  //   this.layer.resetState();

  //   const ys: Tensor1D[] = [];

  //   for (const x of xSequence) {
  //     const y = this.layer.fw(x);
  //     ys.push(y);
  //   }

  //   return ys;
  // }

  // trainOne(xSequence: Tensor1D[], cSequence: Tensor1D[], r: number) {
  //   const ys = this.fw(xSequence);

  //   const ds: Tensor1D[] = [];

  //   for (let t = 0; t < ys.length; t++) {
  //     const d = Tensor.map1D(ys[t], (y, i) => {
  //       const c = cSequence[t][i];
  //       return y - c;
  //     });

  //     ds.push(d);
  //   }

  //   let next_h_d = Tensor.zeros1D(this.layer.hSize);
  //   let curr_y_d = Tensor.zeros1D(this.layer.hSize);

  //   // 時系列の後（next）から前（current）へと遷移
  //   for (let t = xSequence.length - 1; t >= 0; t--) {
  //     curr_y_d = ds[t];
  //     next_h_d = this.layer.bw(curr_y_d, next_h_d, r);
  //   }
  // }

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

  // predict(x: Tensor1D[]): Tensor1D[] {
  //   return this.fw(x);
  // }
}

const trainingSamples = [
  {
    x: [[0.1], [0.2], [0.3], [0.4]], // Increase
    c: [[0.2], [0.3], [0.4], [0.5]], // next value
  },
  {
    x: [[0.5], [0.6], [0.7], [0.8]], // Increase
    c: [[0.6], [0.7], [0.8], [0.9]], // next value
  },
  {
    x: [[0.9], [0.8], [0.7], [0.6]], // Decrease
    c: [[0.8], [0.7], [0.6], [0.5]], // next value
  },
  {
    x: [[0.4], [0.3], [0.2], [0.1]], // Decrease
    c: [[0.3], [0.2], [0.1], [0.0]], // next value
  },
];

const network = new Network(1, 8, 1);

network.trainAll(trainingSamples, 10000, 0.1);

const samples = [
  [[0.1], [0.2], [0.3]], // Unknown Increase Data
  [[0.9], [0.8], [0.7]], // Unknown Decrease Data
];

for (const sample of samples) {
  const y = network.predict(sample);
  console.log(`x: ${sample}, y: ${y[y.length - 1][0].toFixed(4)}`);
}
