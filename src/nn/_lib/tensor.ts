export type Tensor2D = number[][];

export const Tensor = {
  rand(ySize: number, xSize: number, value: number | undefined = 1): Tensor2D {
    return Array.from({ length: ySize }, () =>
      new Array(xSize).fill(0).map(() => Math.random() * (value * 2) - value),
    );
  },

  fill(
    ySize: number,
    xSize: number,
    value: number | undefined = undefined,
  ): Tensor2D {
    return Array.from({ length: ySize }, () =>
      new Array(xSize).fill(value ?? 0),
    );
  },

  T(m: Tensor2D): Tensor2D {
    return m[0].map((_, i) => m.map((row) => row[i]));
  },

  dot(a: Tensor2D, b: Tensor2D): Tensor2D {
    const result = Tensor.fill(a.length, b[0].length, 0); // a.length行 × b[0].length列
    for (let i = 0; i < a.length; i++) {
      for (let k = 0; k < b.length; k++) {
        for (let j = 0; j < b[0].length; j++) {
          result[i][j] += a[i][k] * b[k][j];
        }
      }
    }
    return result;
  },

  mul(a: Tensor2D, b: Tensor2D): Tensor2D {
    return a.map((row, i) => row.map((v, j) => v * b[i][j]));
  },

  add(a: Tensor2D, b: Tensor2D): Tensor2D {
    return a.map((row, i) => row.map((v, j) => v + b[i][j]));
  },

  sub(a: Tensor2D, b: Tensor2D): Tensor2D {
    return a.map((row, i) => row.map((v, j) => v - b[i][j]));
  },

  scale(a: Tensor2D, n: number): Tensor2D {
    return a.map((row) => row.map((v) => v * n));
  },

  map(a: Tensor2D, fn: (v: number) => number): Tensor2D {
    return a.map((row) => row.map(fn));
  },

  sum(a: Tensor2D, axis: 0 | 1): Tensor2D {
    const iH = a.length;
    const iW = a[0].length;

    const result = axis === 0 ? new Array(iW).fill(0) : new Array(iH).fill(0);

    for (let i = 0; i < iH; i++) {
      for (let j = 0; j < iW; j++) {
        if (axis === 0) {
          result[j] += a[i][j];
        } else {
          result[i] += a[i][j];
        }
      }
    }

    return [result];
  },

  clone(a: Tensor2D): Tensor2D {
    return a.map((row) => row.slice());
  },

  rawToConv(x: Tensor2D, kH: number, kW: number): Tensor2D {
    const iH = x.length;
    const iW = x[0].length;
    const oH = iH - kH + 1;
    const oW = iW - kW + 1;

    const result: Tensor2D = [];

    for (let i = 0; i < oH; i++) {
      for (let j = 0; j < oW; j++) {
        const patch: number[] = [];

        for (let ky = 0; ky < kH; ky++) {
          for (let kx = 0; kx < kW; kx++) {
            patch.push(x[i + ky][j + kx]);
          }
        }

        result.push(patch);
      }
    }

    return result;
  },

  convToRaw(
    x: Tensor2D,
    iH: number,
    iW: number,
    kH: number,
    kW: number,
  ): Tensor2D {
    const oH = iH - kH + 1;
    const oW = iW - kW + 1;

    const result = Tensor.fill(iH, iW, 0);

    for (let i = 0; i < oH; i++) {
      for (let j = 0; j < oW; j++) {
        const patch = x[i * oW + j];

        for (let ky = 0; ky < kH; ky++) {
          for (let kx = 0; kx < kW; kx++) {
            result[i + ky][j + kx] += patch[ky * kW + kx];
          }
        }
      }
    }

    return result;
  },
};
