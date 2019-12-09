import {readFileSync} from 'fs';
import {strictEqual} from 'assert';

import {tensor, Tensor, Rank} from '@tensorflow/tfjs';

const uInt = (a: number, b: number, c: number, d: number): number => (a << 24) | (b << 16) | (c << 8) | d;

export interface Set<R extends Rank = Rank> {
  images: Tensor<R>;
  labels: Tensor<R>;
  samples: number;
}

export const load = (width: number, height: number, images: string, labels: string): Set => {
  const imageBlob: Uint8Array = readFileSync(images);
  const labelBlob: Uint8Array = readFileSync(labels);

  const imageSamples: number = uInt(imageBlob[4], imageBlob[5], imageBlob[6], imageBlob[7]);
  const imageWidth: number = uInt(imageBlob[8], imageBlob[9], imageBlob[10], imageBlob[11]);
  const imageHeight: number = uInt(imageBlob[12], imageBlob[13], imageBlob[14], imageBlob[15]);
  const labelSamples: number = uInt(labelBlob[4], labelBlob[5], labelBlob[6], labelBlob[7]);

  strictEqual(imageSamples, labelSamples, `Expected image and label sample count to be equal`);
  strictEqual(imageBlob[2], 8, `Expected image value type to be 8 got ${imageBlob[2]}`);
  strictEqual(imageBlob[3], 3, `Expected image dimension count to be 3 got ${imageBlob[3]}`);
  strictEqual(imageWidth, width, `Expected image width to be ${width} got ${imageWidth}`);
  strictEqual(imageHeight, height, `Expected image width to be ${height} got ${imageHeight}`);

  const imageSize: number = width * height;

  let imageArray: Float32Array = new Float32Array(imageSamples * imageSize);
  let labelArray: Uint8Array = new Uint8Array(labelSamples);

  for (let i: number = 0; i < imageSamples * imageSize; i++) imageArray[i] = imageBlob[16 + i] / 255;
  for (let i: number = 0; i < labelSamples; i++) labelArray[i] = labelBlob[8 + i];

  return {
    images: tensor(imageArray, [imageSamples, width, height, 1]),
    labels: tensor(labelArray, [labelSamples]),
    samples: imageSamples
  };
};
