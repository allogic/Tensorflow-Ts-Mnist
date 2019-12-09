import {Sequential, split} from '@tensorflow/tfjs';

import {WIDTH, HEIGHT, load} from './data';
import {cnn, train} from './model';

const [trainingSamples, trainingImages, trainingLabels] = load('./train-images-idx3-ubyte', './train-labels-idx1-ubyte');
const [testSamples, testImages, testLabels] = load('./t10k-images-idx3-ubyte', './t10k-labels-idx1-ubyte');

train(cnn(), trainingImages, trainingLabels).then(() => {
  console.log('finished');
}).catch(error => console.log(error));

const [a] = split(trainingImages, trainingSamples);
const [b] = split(trainingLabels, trainingSamples);

b.print();

for (let i: number = 0; i < 28; i++) {
  let line: string = '';

  for (let j: number = 0; j < 28; j++)
    line += a.bufferSync().get(0, i, j) === 0 ? '0' : '1';

  console.log(line);
}
