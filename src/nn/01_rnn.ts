import { type Activation, Activations } from "./_lib/activation";
import { Tensor, type Tensor2D } from "./_lib/tensor";

class Layer {
  xSize: number;
  hSize: number;
  ySize: number;

  xs: Tensor2D[] = [];
  as: Tensor2D[] = [];
  hs: Tensor2D[] = [];
  ys: Tensor2D[] = [];

  hCurr: Tensor2D;

  wX: Tensor2D;
  wH: Tensor2D;
  wY: Tensor2D;

  bH: Tensor2D;
  bY: Tensor2D;

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

    this.wX = Tensor.rand(xSize, hSize);
    this.wH = Tensor.rand(hSize, hSize);
    this.wY = Tensor.rand(hSize, ySize);

    this.bH = Tensor.fill(1, hSize, 0);
    this.bY = Tensor.fill(1, ySize, 0);

    this.hCurr = Tensor.fill(1, hSize, 0);
  }

  fw(currX: Tensor2D): Tensor2D {
    const hPrev = this.hCurr;

    this.xs.push(currX);

    // h_{t-1} * h_w
    // x       * x_w
    const z = Tensor.add(
      Tensor.add(Tensor.dot(hPrev, this.wH), Tensor.dot(currX, this.wX)),
      this.bH,
    );

    const h = this.activation.fw(z);

    this.hs.push(h);

    this.hCurr = Tensor.clone(h);

    const y = Tensor.add(Tensor.dot(this.hCurr, this.wY), this.bY);

    this.ys.push(y);

    return y;
  }

  bw(dys: Tensor2D[]): {
    gwX: Tensor2D;
    gwH: Tensor2D;
    gwY: Tensor2D;
    gbH: Tensor2D;
    gbY: Tensor2D;
  } {
    let gwX = Tensor.fill(this.xSize, this.hSize, 0);
    let gwH = Tensor.fill(this.hSize, this.hSize, 0);
    let gwY = Tensor.fill(this.hSize, this.ySize, 0);
    let gbH = Tensor.fill(1, this.hSize, 0);
    let gbY = Tensor.fill(1, this.ySize, 0);

    // 勾配 h_{t+1}
    let ghNext = Tensor.fill(1, this.hSize, 0);

    for (let t = this.xs.length - 1; t >= 0; t--) {
      const xCurr = this.xs[t];
      const hCurr = this.hs[t];

      const dy = dys[t];

      // h_{t-1}
      let hPrev: Tensor2D;
      if (t > 0) {
        hPrev = this.hs[t - 1];
      } else {
        hPrev = Tensor.fill(1, this.hSize, 0);
      }

      const dh = Tensor.add(Tensor.dot(dy, Tensor.T(this.wY)), ghNext);

      const dz = Tensor.mul(dh, this.activation.bw(hCurr));

      gwX = Tensor.add(gwX, Tensor.dot(Tensor.T(xCurr), dz));
      gwH = Tensor.add(gwH, Tensor.dot(Tensor.T(hPrev), dz));
      gwY = Tensor.add(gwY, Tensor.dot(Tensor.T(hCurr), dy));
      gbH = Tensor.add(gbH, dz);
      gbY = Tensor.add(gbY, dy);

      ghNext = Tensor.dot(dz, Tensor.T(this.wH));
    }

    return { gwX, gwH, gwY, gbH, gbY };
  }

  update(
    grads: {
      gwX: Tensor2D;
      gwH: Tensor2D;
      gwY: Tensor2D;
      gbH: Tensor2D;
      gbY: Tensor2D;
    },
    lr: number,
  ) {
    this.wX = Tensor.sub(this.wX, Tensor.scale(grads.gwX, lr));
    this.wH = Tensor.sub(this.wH, Tensor.scale(grads.gwH, lr));
    this.wY = Tensor.sub(this.wY, Tensor.scale(grads.gwY, lr));
    this.bH = Tensor.sub(this.bH, Tensor.scale(grads.gbH, lr));
    this.bY = Tensor.sub(this.bY, Tensor.scale(grads.gbY, lr));
  }

  resetState() {
    this.xs = [];
    this.hs = [];
    this.ys = [];
    this.hCurr = Tensor.fill(1, this.hSize, 0);
  }
}

class Network {
  layer: Layer;

  constructor(xSize: number, hSize: number, ySize: number) {
    this.layer = new Layer(xSize, hSize, ySize, Activations.sigmoid);
  }

  fw(xSequence: Tensor2D[]): Tensor2D[] {
    this.layer.resetState();

    const ys: Tensor2D[] = [];

    for (const x of xSequence) {
      ys.push(this.layer.fw(x));
    }

    return ys;
  }

  trainOne(xSeq: Tensor2D[], tSeq: Tensor2D[], lr: number) {
    const ys = this.fw(xSeq);

    const gys: Tensor2D[] = ys.map((y, t) => Tensor.sub(y, tSeq[t]));

    const gs = this.layer.bw(gys);

    this.layer.update(this.clipGrads(gs, 1.0), lr);

    this.layer.resetState();
  }

  trainAll(
    data: { x: Tensor2D[]; t: Tensor2D[] }[],
    epochs: number,
    lr: number,
  ) {
    for (let i = 0; i < epochs; i++) {
      let epochLoss = 0;

      for (const sample of data) {
        const ys = this.fw(sample.x);

        ys.forEach((y, t) => {
          const diff = Tensor.sub(y, sample.t[t]);
          epochLoss += diff[0][0] ** 2;
        });

        this.trainOne(sample.x, sample.t, lr);
      }

      if (i % 1000 === 0) {
        console.log(`Epoch ${i}: Loss ${epochLoss.toFixed(6)}`);
      }
    }
  }

  predict(x: Tensor2D[]): Tensor2D[] {
    return this.fw(x);
  }

  private clipGrads(
    grads: {
      gwX: Tensor2D;
      gwH: Tensor2D;
      gwY: Tensor2D;
      gbH: Tensor2D;
      gbY: Tensor2D;
    },
    threshold: number,
  ) {
    const clip = (t: Tensor2D) => {
      return Tensor.map(t, (val) =>
        Math.max(Math.min(val, threshold), -threshold),
      );
    };

    return {
      gwX: clip(grads.gwX),
      gwH: clip(grads.gwH),
      gwY: clip(grads.gwY),
      gbH: clip(grads.gbH),
      gbY: clip(grads.gbY),
    };
  }
}

const trainingSamples = [
  {
    x: [[[0.1]], [[0.2]], [[0.3]], [[0.4]]], // Increase
    t: [[[0.2]], [[0.3]], [[0.4]], [[0.5]]], // next value
  },
  {
    x: [[[0.5]], [[0.6]], [[0.7]], [[0.8]]], // Increase
    t: [[[0.6]], [[0.7]], [[0.8]], [[0.9]]], // next value
  },
  {
    x: [[[0.9]], [[0.8]], [[0.7]], [[0.6]]], // Decrease
    t: [[[0.8]], [[0.7]], [[0.6]], [[0.5]]], // next value
  },
  {
    x: [[[0.4]], [[0.3]], [[0.2]], [[0.1]]], // Decrease
    t: [[[0.3]], [[0.2]], [[0.1]], [[0.0]]], // next value
  },
];

const network = new Network(1, 8, 1);

network.trainAll(trainingSamples, 10000, 0.1);

const samples = [
  [[[0.3]], [[0.4]], [[0.5]]], // Unknown Increase Data
  [[[0.7]], [[0.6]], [[0.5]]], // Unknown Decrease Data
];

for (const sample of samples) {
  const y = network.predict(sample);
  console.log(`x: ${sample}, y: ${y[y.length - 1][0][0].toFixed(4)}`);
}
