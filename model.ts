import {layers, sequential, Sequential, Tensor, nextFrame} from '@tensorflow/tfjs-node';

import {WIDTH, HEIGHT} from './data';

export const cnn = (): Sequential => {
  const model: Sequential = sequential();

  model.add(layers.conv2d({
    inputShape: [WIDTH, HEIGHT, 1],
    kernelSize: 3,
    filters: 16,
    activation: 'relu',
  }));

  model.add(layers.maxPool2d({
    poolSize: 2,
    strides: 2
  }));

  model.add(layers.conv2d({
    kernelSize: 3,
    filters: 32,
    activation: 'relu'
  }));

  model.add(layers.maxPool2d({
    poolSize: 2,
    strides: 2
  }));

  model.add(layers.conv2d({
    kernelSize: 3,
    filters: 32,
    activation: 'relu'
  }));

  model.add(layers.flatten());

  model.add(layers.dense({
    units: 64,
    activation: 'relu'
  }));

  model.add(layers.dense({
    units: 10,
    activation: 'softmax'
  }));

  return model;
};
