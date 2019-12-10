import {load} from './data';
import {cnn} from './model';

import {Tensor, Sequential, split} from '@tensorflow/tfjs-node';

const [trainingImages, trainingLabels] = load('./train-images-idx3-ubyte', './train-labels-idx1-ubyte');
const [testImages, testLabels] = load('./t10k-images-idx3-ubyte', './t10k-labels-idx1-ubyte');

const render = (image: Tensor, label: Tensor): void => {
  label.print();

  for (let i: number = 0; i < 28; i++) {
    let line: string = '';

    for (let j: number = 0; j < 28; j++)
      line += image.bufferSync().get(0, i, j) === 0 ? '0' : '1';

    console.log(line);
  }
};

const model: Sequential = cnn();

const train = async (): Promise<void> => {
  model.compile({
    optimizer: 'rmsprop',
    loss: 'meanSquaredError',
    metrics: ['acc']
  });

  model.summary();

  await model.fit(trainingImages, trainingLabels, {
    shuffle: true,
    batchSize: 6000,
    epochs: 10,
    validationData: [testImages, testLabels],
  });
};

train().then(() => {
  const predictions: any = model.predict(testImages);

  const xs = split(testImages, 10000);
  const ys = split(predictions, 10000);

  render(xs[0], ys[0]);
  render(xs[1], ys[1]);
  render(xs[2], ys[2]);
  render(xs[3], ys[3]);
});
