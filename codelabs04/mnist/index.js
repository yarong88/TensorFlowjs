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

// MNIST 데이터를 로딩하고 관리하기 위한 헬퍼 클래스
import { IMAGE_H, IMAGE_W, MnistData } from "./data.js";

// 손실 그래프와 MNIST 이미지를 그리기 위한 헬퍼 클래스
import * as ui from "./ui.js";

/**
 * 합성곱 신경망을 만듭니다.
 */

/* 학생 실습 코드삽입 */
// 합성곱 신경망의 첫 번째 층은 28X28 픽셀의 흑백 이미지를 받음
// 이층은 커널 크기가 3필셀인 16개 필터를 사용하고 렐루 활성화 함수를 사용함
function createConvModel() {
  const model = tf.sequential();
  model.add(
    tf.layers.conv2d({
      inputShape: [IMAGE_H, IMAGE_W, 1],
      kernelSize: 3,
      filters: 16,
      activation: "relu",
    })
  );
  // 첫 번째 층 다음에 최대 풀링층을 둔다.
  // 주어진 영역에서 최대값을 고르며 다운 샘플링처럼 동작한다. (폴링의 종류에는 max나 average 평균 등 많다.)
  // flatten
  model.add(tf.layers.maxPooling2d({ poolSize: 2, strides: 2 }));
  //poolSize => pooling하는 사이즈(몇번 pooling하는가에 따라 사이즈가 정해짐(이동))

  // 세번째 층은 다른 합성곱 층, 이번에는 32개의 필터를 사용
  model.add(
    tf.layers.conv2d({ kernelSize: 3, filters: 32, activation: "relu" })
  );
  // 풀링층 추가
  model.add(tf.layers.maxPooling2d({ poolSize: 2, strides: 2 }));
  // 세번째 conv 추가
  model.add(
    tf.layers.conv2d({ kernelSize: 3, filters: 32, activation: "relu" })
  );
  // 마지막 층에 입력으로 전달하기 위해 3D 텐서를 1D 백터로 펼침
  // 마지막 분류중에 고차원 데이처를 전달하기 위해 자주 사용하는 기법
  model.add(tf.layers.flatten({}));

  model.add(tf.layers.dense({ units: 64, activation: "relu" }));

  model.add(tf.layers.dense({ units: 10, activation: "softmax" }));
  return model;
}

/**
 * 아래 flatten, dense 층만 가진 모델을 만듭니다.
 *
 * 이 모델은 `createConvModel()`로 만든 합성곱 신경망과 거의 같은 개수(~31k)의 파라미터를 가집니다.
 * 하지만 합성곱 신경망처럼 공간 정보를 사용하지 못하기 때문에 훈련 정확도는 크게 떨어집니다.
 *
 * 합성곱 신경망과 비교를 위한 모델입니다.
 *
 * @returns {tf.Model} tf.Model 객체
 */
function createDenseModel() {
  const model = tf.sequential();
  model.add(tf.layers.flatten({ inputShape: [IMAGE_H, IMAGE_W, 1] }));
  model.add(tf.layers.dense({ units: 42, activation: "relu" }));
  model.add(tf.layers.dense({ units: 10, activation: "softmax" }));
  return model;
}

//모델을 컴파일하고 훈련합니다.

async function train(model, onIteration) {
  ui.logStatus("모델을 훈련합니다...");

  const optimizer = "rmsprop";
  model.compile({
    optimizer,
    loss: "categoricalCrossentropy",
    metrics: ["accuracy"],
  });

  const batchSize = 64;

  const validationSplit = 0.15;

  const trainEpochs = ui.getTrainEpochs();

  let trainBatchCount = 0;

  const trainData = data.getTrainData();
  const testData = data.getTestData();

  const totalNumBatches =
    Math.ceil((trainData.xs.shape[0] * (1 - validationSplit)) / batchSize) *
    trainEpochs;

  let valAcc;
  await model.fit(trainData.xs, trainData.labels, {
    batchSize: batchSize,
    validationSplit: validationSplit,
    epochs: trainEpochs,
    callbacks: {
      onBatchEnd: async (batch, logs) => {
        trainBatchCount++;
        ui.logStatus(
          `훈련... (` +
            `${((trainBatchCount / totalNumBatches) * 100).toFixed(1)}%` +
            ` 완료). 훈련을 멈추려면 페이지를 새로고침하거나 종료하세요.`
        );
        ui.plotLoss(trainBatchCount, logs.loss, "train");
        ui.plotAccuracy(trainBatchCount, logs.acc, "train");
        if (onIteration && batch % 10 === 0) {
          onIteration("onBatchEnd", batch, logs);
        }
        await tf.nextFrame();
      },
      onEpochEnd: async (epoch, logs) => {
        valAcc = logs.val_acc;
        ui.plotLoss(trainBatchCount, logs.val_loss, "validation");
        ui.plotAccuracy(trainBatchCount, logs.val_acc, "validation");
        if (onIteration) {
          onIteration("onEpochEnd", epoch, logs);
        }
        await tf.nextFrame();
      },
    },
  });

  const testResult = model.evaluate(testData.xs, testData.labels);
  const testAccPercent = testResult[1].dataSync()[0] * 100;
  const finalValAccPercent = valAcc * 100;
  ui.logStatus(
    `최종 검증 정확도: ${finalValAccPercent.toFixed(1)}%; ` +
      `최종 테스트 정확도: ${testAccPercent.toFixed(1)}%`
  );
}

//테스트 샘플에 대한 예측을 출력합니다.

async function showPredictions(model) {
  const testExamples = 100;
  const examples = data.getTestData(testExamples);

  // dispose()를 호출하지 않고 실행 후에 GPU 메모리에서 텐서를 해제하기 위해 tf.tidy()를 사용합니다.
  // tf.tidy는 동기적으로 실행됩니다.
  tf.tidy(() => {
    const output = model.predict(examples.xs);

    // tf.argMax()는 특정 축을 따라 텐서에서 가장 큰 값의 인덱스를 반환합니다.
    // 이 예제와 같은 다중 분류 작업은 종종 클래스를 원-핫 벡터로 표현합니다.
    // 원-핫 벡터는 클래스마다 하나의 원소로 이루어진 1D 벡터입니다.
    // 이 벡터의 원소는 하나만 1이고 모두 0입니다(예를 들면, [0, 0, 0, 1, 0]).
    // model.predict()는 확률 분포를 출력합니다. 따라서 argMax를 사용해 가장 높은 확률을 가진 원소의 인덱스를 찾습니다.
    // 이 값이 예측이 됩니다. (예를 들어, argmax([0.07, 0.1, 0.03, 0.75, 0.05]) == 3)
    // dataSync()는 일반 CPU 자바스크립트 코드에서 사용하기 위해 GPU에서 tf.tensor 값을 동기적으로 다운로드합니다.
    // (이 함수의 비동기 버전은 data()입니다)
    const axis = 1;
    const labels = Array.from(examples.labels.argMax(axis).dataSync());
    const predictions = Array.from(output.argMax(axis).dataSync());

    ui.showTestResults(examples, predictions, labels);
  });
}

function createModel() {
  let model;
  const modelType = ui.getModelTypeId();
  if (modelType === "ConvNet") {
    model = createConvModel();
  } else if (modelType === "DenseNet") {
    model = createDenseModel();
  } else {
    throw new Error(`잘못된 모델 종류입니다: ${modelType}`);
  }
  return model;
}

let data;
async function load() {
  data = new MnistData();
  await data.load();
}

ui.setTrainButtonCallback(async () => {
  ui.logStatus("MNIST 데이터를 로딩합니다...");
  await load();

  ui.logStatus("모델을 만듭니다...");
  const model = createModel();
  model.summary();

  ui.logStatus("모델 훈련을 시작합니다...");
  await train(model, () => showPredictions(model));
});
