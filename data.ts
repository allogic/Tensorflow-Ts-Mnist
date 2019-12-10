import {readFileSync} from 'fs';
import {strictEqual} from 'assert';

import {buffer, Tensor} from '@tensorflow/tfjs-node';

const uInt = (a: number, b: number, c: number, d: number): number => (a << 24) | (b << 16) | (c << 8) | d;

export const WIDTH: number = 28;
export const HEIGHT: number = 28;

const SIZE: number = WIDTH * HEIGHT;

export const load = (images: string, labels: string): Tensor[] => {
  const imageBlob: Uint8Array = readFileSync(images);
  const labelBlob: Uint8Array = readFileSync(labels);

  const imageSamples: number = uInt(imageBlob[4], imageBlob[5], imageBlob[6], imageBlob[7]);
  const labelSamples: number = uInt(labelBlob[4], labelBlob[5], labelBlob[6], labelBlob[7]);

  const imageWidth: number = uInt(imageBlob[8], imageBlob[9], imageBlob[10], imageBlob[11]);
  const imageHeight: number = uInt(imageBlob[12], imageBlob[13], imageBlob[14], imageBlob[15]);

  strictEqual(imageSamples, labelSamples, `Expected image and label sample count to be equal`);

  strictEqual(imageBlob[2], 8, `Expected image value type to be 8 got ${imageBlob[2]}`);
  strictEqual(labelBlob[2], 8, `Expected label value type to be 8 got ${labelBlob[2]}`);

  strictEqual(imageBlob[3], 3, `Expected image dimension count to be 3 got ${imageBlob[3]}`);
  strictEqual(labelBlob[3], 1, `Expected image dimension count to be 1 got ${labelBlob[3]}`);

  strictEqual(imageWidth, WIDTH, `Expected image width to be ${WIDTH} got ${imageWidth}`);
  strictEqual(imageHeight, HEIGHT, `Expected image width to be ${HEIGHT} got ${imageHeight}`);

  const imageBuffer = buffer([imageSamples * SIZE]);
  const labelBuffer = buffer([labelSamples, 10]);

  for (let i: number = 0; i < imageSamples * SIZE; i++) imageBuffer.set(imageBlob[16 + i] / 255, i);
  for (let i: number = 0; i < labelSamples; i++) labelBuffer.set(1, i, labelBlob[8 + i]);

  return [
    imageBuffer.toTensor().reshape([imageSamples, WIDTH, HEIGHT, 1]),
    labelBuffer.toTensor()
  ];
};
