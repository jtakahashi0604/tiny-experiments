import { type Activation, Activations } from "./_lib/activation";
import { FlattenLayer, LinearLayer } from "./_lib/layer";
import { Tensor, type Tensor2D } from "./_lib/tensor";

class Layer {
  iH: number;
  iW: number;
  oH: number;
  oW: number;
  kH: number;
  kW: number;
  filterSize: number;

  x: Tensor2D; // [iH    * iW]
  c: Tensor2D; // [oH*oW * kH*kW]
  y: Tensor2D; // [oH*oW * filterSize]
  w: Tensor2D; // [kH*kW * filterSize]
  b: Tensor2D; // [1     * filterSize]

  activation: Activation;

  constructor(
    iH: number,
    iW: number,
    kH: number,
    kW: number,
    filterSize: number,
    activation: Activation,
  ) {
    this.iH = iH;
    this.iW = iW;
    this.kH = kH;
    this.kW = kW;
    this.oH = iH - kH + 1;
    this.oW = iW - kW + 1;
    this.filterSize = filterSize;

    this.activation = activation;

    this.w = Tensor.rand(kH * kW, filterSize);
    this.b = Tensor.fill(1, filterSize, 0);
  }

  fw(x: Tensor2D): Tensor2D {
    const c = Tensor.rawToConv(x, this.kH, this.kW);

    let z = Tensor.dot(c, this.w);

    z = z.map((row) => row.map((v, i) => v + this.b[0][i]));

    const a = this.activation.fw(z);

    this.x = x;
    this.c = c;
    this.y = a;

    return this.y;
  }

  bw(gy: Tensor2D, lr: number): Tensor2D {
    const gz = Tensor.mul(gy, this.activation.bw(this.y));

    const gc = Tensor.dot(gz, Tensor.T(this.w));
    const gw = Tensor.dot(Tensor.T(this.c), gz);

    const gb = Tensor.sum(gz, 0);

    this.w = Tensor.sub(this.w, Tensor.scale(gw, lr));
    this.b = Tensor.sub(this.b, Tensor.scale(gb, lr));

    const gx = Tensor.convToRaw(gc, this.iH, this.iW, this.kH, this.kW);

    return gx;
  }
}

type AnyLayer = Layer | FlattenLayer | LinearLayer;

class Network {
  layers: AnyLayer[];

  constructor(
    iH: number,
    iW: number,
    kH: number,
    kW: number,
    filterSize: number,
  ) {
    const oH = iH - kH + 1;
    const oW = iW - kW + 1;

    const flattenSize = oH * oW * filterSize;

    this.layers = [
      new Layer(iH, iW, kH, kW, filterSize, Activations.sigmoid),
      new FlattenLayer(),
      new LinearLayer(flattenSize, 1, Activations.sigmoid),
    ];
  }

  fw(x: Tensor2D): Tensor2D {
    return this.layers.reduce((currX, layer) => {
      return layer.fw(currX) as Tensor2D;
    }, x);
  }

  trainOne(x: Tensor2D, t: Tensor2D, lr: number) {
    const y = this.fw(x);

    let currGy = Tensor.sub(y, t);

    for (let i = this.layers.length - 1; i >= 0; i--) {
      const layer = this.layers[i];

      if (layer instanceof FlattenLayer) {
        currGy = layer.bw(currGy);
      } else {
        currGy = layer.bw(currGy, lr);
      }
    }
  }

  trainAll(data: { x: Tensor2D; t: Tensor2D }[], epochs: number, lr: number) {
    for (let i = 0; i < epochs; i++) {
      for (const sample of data) {
        this.trainOne(sample.x, sample.t, lr);
      }
    }
  }

  predict(x: Tensor2D): Tensor2D {
    return this.fw(x);
  }
}

const trainingSamples = [
  // 横線
  {
    x: [
      [1, 1, 1],
      [0, 0, 0],
      [0, 0, 0],
    ],
    t: [[1.0]],
  }, // 上
  {
    x: [
      [0, 0, 0],
      [1, 1, 1],
      [0, 0, 0],
    ],
    t: [[1.0]],
  }, // 中
  {
    x: [
      [0, 0, 0],
      [0, 0, 0],
      [1, 1, 1],
    ],
    t: [[1.0]],
  }, // 下

  // 縦線
  {
    x: [
      [1, 0, 0],
      [1, 0, 0],
      [1, 0, 0],
    ],
    t: [[0.0]],
  }, // 左
  {
    x: [
      [0, 1, 0],
      [0, 1, 0],
      [0, 1, 0],
    ],
    t: [[0.0]],
  }, // 中
  {
    x: [
      [0, 0, 1],
      [0, 0, 1],
      [0, 0, 1],
    ],
    t: [[0.0]],
  }, // 右
];

const network = new Network(3, 3, 2, 2, 2);

network.trainAll(trainingSamples, 1000, 0.2);

const samples = [
  // 横線
  [
    [0, 0, 0],
    [0, 0, 0],
    [1, 1, 1],
  ],
  // 縦線
  [
    [0, 0, 1],
    [0, 0, 1],
    [0, 0, 1],
  ],
];

for (const sample of samples) {
  const y = network.predict(sample);
  console.log(`x: ${sample}, y: ${y[0][0].toFixed(4)}`);
}
