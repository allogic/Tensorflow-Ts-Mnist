import {layers, sequential, Sequential} from "@tensorflow/tfjs";

export const convolutional = (width: number, height: number): Sequential => {
  const model: Sequential = sequential();

  model.add(layers.conv2d({
    inputShape: [width, height, 1],
    kernelSize: 3,
    filters: 16,
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
