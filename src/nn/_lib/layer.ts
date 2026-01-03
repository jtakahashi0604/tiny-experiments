import type { Activation } from "./activation";
import { Tensor, type Tensor2D } from "./tensor";

export class LinearLayer {
  xSize: number;
  ySize: number;

  x: Tensor2D; // [1     * xSize]
  y: Tensor2D; // [1     * ySize]
  w: Tensor2D; // [xSize * ySize]
  b: Tensor2D; // [1     * ySize]

  activation: Activation;

  constructor(xSize: number, ySize: number, activation: Activation) {
    this.xSize = xSize;
    this.ySize = ySize;
    this.activation = activation;

    this.w = Tensor.rand(xSize, ySize);
    this.b = Tensor.fill(1, ySize, 0);
  }

  fw(x: Tensor2D): Tensor2D {
    const z = Tensor.add(Tensor.dot(x, this.w), this.b);
    const a = this.activation.fw(z);

    this.x = x;
    this.y = a;

    return this.y;
  }

  bw(gy: Tensor2D, lr: number): Tensor2D {
    const gz = Tensor.mul(gy, this.activation.bw(this.y));

    const gx = Tensor.dot(gz, Tensor.T(this.w));
    const gw = Tensor.dot(Tensor.T(this.x), gz);

    this.w = Tensor.sub(this.w, Tensor.scale(gw, lr));
    this.b = Tensor.sub(this.b, Tensor.scale(gz, lr));

    return gx;
  }
}

export class FlattenLayer {
  private xShape: [number, number] = [0, 0];

  fw(x: Tensor2D): Tensor2D {
    this.xShape = [x.length, x[0].length];

    const result = [x.reduce((acc, row) => acc.concat(row), [])];

    return result;
  }

  bw(gy: Tensor2D): Tensor2D {
    const [rows, cols] = this.xShape;

    const result: Tensor2D = [];

    for (let i = 0; i < rows; i++) {
      result.push(gy[0].slice(i * cols, (i + 1) * cols));
    }

    return result;
  }
}
