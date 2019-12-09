import {Sequential} from '@tensorflow/tfjs';

import {load, Set} from './data';
import {convolutional} from './model';

const width: number = 28;
const height: number = 28;

const training: Set = load(width, height, './train-images.idx3-ubyte', './train-labels.idx1-ubyte');
const test: Set = load(width, height, 't10k-images.idx3-ubyte', 't10k-labels.idx1-ubyte');

const model: Sequential = convolutional(width, height);



/*
const [av, bv] = split(training.images, training.samples);
const [al, bl] = split(training.labels, training.samples);

al.print();

for (let i: number = 0; i < 28; i++) {
  let line: string = '';

  for (let j: number = 0; j < 28; j++)
    line += av.bufferSync().get(0, i, j) === 0 ? '0' : '1';

  console.log(line);
}
*/
