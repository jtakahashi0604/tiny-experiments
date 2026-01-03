import { type Activation, Activations } from "./_lib/activation";
import { LinearLayer } from "./_lib/layer";
import { Tensor, type Tensor2D } from "./_lib/tensor";

class Network {
  layers: LinearLayer[];

  constructor(xSize: number, hSize: number, ySize: number) {
    this.layers = [
      new LinearLayer(xSize, hSize, Activations.sigmoid),
      new LinearLayer(hSize, ySize, Activations.sigmoid),
    ];
  }

  fw(x: Tensor2D): Tensor2D {
    return this.layers.reduce((currX, layer) => {
      return layer.fw(currX);
    }, x);
  }

  trainOne(x: Tensor2D, t: Tensor2D, lr: number) {
    const y = this.fw(x);

    let currGy = Tensor.sub(y, t);

    for (let i = this.layers.length - 1; i >= 0; i--) {
      currGy = this.layers[i].bw(currGy, lr);
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
  { x: [[0, 0]], t: [[0]] },
  { x: [[0, 1]], t: [[1]] },
  { x: [[1, 0]], t: [[1]] },
  { x: [[1, 1]], t: [[0]] },
];

const network = new Network(2, 2, 1);

network.trainAll(trainingSamples, 10000, 0.5);

const samples = [[[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]]];

for (const sample of samples) {
  const y = network.predict(sample);

  console.log(`x: ${sample}, y: ${y[0][0].toFixed(4)}`);
}
