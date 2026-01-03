import { Tensor, type Tensor2D } from "./tensor";

export interface Activation {
  fw(z: Tensor2D): Tensor2D;
  bw(a: Tensor2D): Tensor2D;
}

export const Activations: { [key: string]: Activation } = {
  sigmoid: {
    fw(z: Tensor2D): Tensor2D {
      /// NOTE: Why using exp?
      // ロジット
      // e
      return Tensor.map(z, (v) => 1 / (1 + Math.exp(-v)));
    },
    bw(h: Tensor2D): Tensor2D {
      return Tensor.map(h, (v) => v * (1 - v));
    },
  },
};
