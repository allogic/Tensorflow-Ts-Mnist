import {load} from './data';
import {cnn} from './model';

const [trainingSamples, trainingImages, trainingLabels] = load('./train-images-idx3-ubyte', './train-labels-idx1-ubyte');
const [testSamples, testImages, testLabels] = load('./t10k-images-idx3-ubyte', './t10k-labels-idx1-ubyte');

/*const t = split(trainingImages, trainingSamples);
const l = split(trainingLabels, trainingSamples);

l[666].print();

for (let i: number = 0; i < 28; i++) {
  let line: string = '';

  for (let j: number = 0; j < 28; j++)
    line += t[666].bufferSync().get(0, i, j) === 0 ? '0' : '1';

  console.log(line);
}*/

const model = cnn();

const train = async () => {
  model.compile({
    optimizer: 'adam',
    loss: 'meanSquaredError',
    metrics: ['acc']
  });

  await model.fit(trainingSamples, trainingLabels, {
    shuffle: true,
    batchSize: 6000,
    epochs: 100
  });
};

train(, trainingImages, trainingLabels).then(() => {

}).catch(error => console.log(error));
