/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

export const IRIS_CLASSES = [
  "Iris-setosa",
  "Iris-versicolor",
  "Iris-virginica",
];
export const IRIS_NUM_CLASSES = IRIS_CLASSES.length;

/* 다음 아이리스 데이터를 넣으시오. */
//  [5.1, 3.5, 1.4, 0.2, 0] ..... 등으로 넣을것
// 맨마지막 숫자 는 setosa = 0 , versicolor = 1, virginica = 2
const IRIS_DATA = [];
import iris from "./iris.json" assert { type: "json" };
for (let i = 0; i < iris.length; i++) {
  let data = [];
  data.push(iris[i].sepalLength);
  data.push(iris[i].sepalWidth);
  data.push(iris[i].petalLength);
  data.push(iris[i].petalWidth);
  if (iris[i].species == "setosa") {
    data.push(0);
  } else if (iris[i].species == "versicolor") {
    data.push(1);
  } else if (iris[i].species == "virginica") {
    data.push(2);
  }
  IRIS_DATA.push(data);
}

/**
 * 붓꽃 데이터 배열을 `tf.Tensor`로 변환합니다.
 * @param data 붓꽃의 입력 특성 데이터. `Array`의 `Array`로 각 원소는 길이가 4인 `Array`입니다.
 *   (꽃잎 길이, 꽃잎 너비, 꽃받침 길이, 꽃받침 너비)
 * @param targets {0, 1, 2} 값으로 이루어진 숫자 `Array`로 붓꽃의 진짜 클래스를 나타냅니다.
 *   `data`와 배열의 길이가 같다고 가정합니다.
 * @param testSplit 테스트 데이터로 분할할 비율: 0과 1 사이 숫자
 * @return 길이가 4인 `Array`
 *   - [numTrainExapmles, 4] 크기의 `tf.Tensor`인 훈련 데이터
 *   - [numTrainExamples, 3] 크기의 `tf.Tensor`인 원-핫 인코딩된 훈련 데이터의 레이블
 *   - [numTestExamples, 4] 크기의 `tf.Tensor`인 테스트 데이터
 *   - [numTestExamples, 3] 크기의 `tf.Tensor`인 원-핫 인코딩된 테스트 데이터의 레이블
 */
function convertToTensors(data, targets, testSplit) {
  const numExamples = data.length;
  if (numExamples !== targets.length) {
    throw new Error("데이터와 타깃의 길이가 다릅니다.");
  }

  // `data`와 `targets`을 랜덤하게 섞습니다.
  const indices = [];
  for (let i = 0; i < numExamples; ++i) {
    indices.push(i);
  }
  tf.util.shuffle(indices);

  const shuffledData = [];
  const shuffledTargets = [];
  for (let i = 0; i < numExamples; ++i) {
    shuffledData.push(data[indices[i]]);
    shuffledTargets.push(targets[indices[i]]);
  }

  // `testSplit`를 기준으로 데이터를 훈련 세트와 테스트 세트로 나눕니다.
  const numTestExamples = Math.round(numExamples * testSplit);
  const numTrainExamples = numExamples - numTestExamples;

  const xDims = shuffledData[0].length;

  // 특성 데이터를 담은 2D `tf.Tensor`를 만듭니다.
  const xs = tf.tensor2d(shuffledData, [numExamples, xDims]);

  // 레이블을 담은 1D `tf.Tensor`를 만들고, 숫자 레이블 {0, 1, 2}을
  // 원-핫 인코딩으로 바꿉니다(예를 들어, 0 --> [1, 0, 0]).
  const ys = tf.oneHot(tf.tensor1d(shuffledTargets).toInt(), IRIS_NUM_CLASSES);
  // `targets`과 `ys` 값을 출력하는 라인 추가.
  console.log("targets 값:", targets);
  ys.print();

  // `slice` 메서드를 사용해 데이터를 훈련 세트와 테스트 세트로 나눕니다.
  const xTrain = xs.slice([0, 0], [numTrainExamples, xDims]);
  const xTest = xs.slice([numTrainExamples, 0], [numTestExamples, xDims]);
  const yTrain = ys.slice([0, 0], [numTrainExamples, IRIS_NUM_CLASSES]);
  const yTest = ys.slice([0, 0], [numTestExamples, IRIS_NUM_CLASSES]);
  return [xTrain, yTrain, xTest, yTest];
}

/**
 * 붓꽃 데이터를 훈련 세트와 테스트 세트로 나눕니다.
 *
 * @param testSplit 테스트 데이터로 분할할 비율: 0과 1 사이 숫자
 *
 * @return 길이가 4인 `Array`
 *   - 훈련 데이터: 길이가 4인 숫자 `Array`의 `Array`
 *   - 훈련 레이블: 훈련 데이터와 길이가 같은 숫자 `Array`. 이 `Array`의 각 원소는 {0, 1, 2} 중 하나입니다.
 *   - 테스트 데이터: 길이가 4인 숫자 `Array`의 `Array`
 *   - 테스트 레이블: 테스트 데이터와 길이가 같은 숫자 `Array`. 이 `Array`의 각 원소는 {0, 1, 2} 중 하나입니다.
 */
export function getIrisData(testSplit) {
  return tf.tidy(() => {
    const dataByClass = [];
    const targetsByClass = [];
    for (let i = 0; i < IRIS_CLASSES.length; ++i) {
      dataByClass.push([]);
      targetsByClass.push([]);
    }
    for (const example of IRIS_DATA) {
      const target = example[example.length - 1];
      const data = example.slice(0, example.length - 1);
      dataByClass[target].push(data);
      targetsByClass[target].push(target);
    }

    const xTrains = [];
    const yTrains = [];
    const xTests = [];
    const yTests = [];
    for (let i = 0; i < IRIS_CLASSES.length; ++i) {
      const [xTrain, yTrain, xTest, yTest] = convertToTensors(
        dataByClass[i],
        targetsByClass[i],
        testSplit
      );
      xTrains.push(xTrain);
      yTrains.push(yTrain);
      xTests.push(xTest);
      yTests.push(yTest);
    }

    const concatAxis = 0;
    return [
      tf.concat(xTrains, concatAxis),
      tf.concat(yTrains, concatAxis),
      tf.concat(xTests, concatAxis),
      tf.concat(yTests, concatAxis),
    ];
  });
}
