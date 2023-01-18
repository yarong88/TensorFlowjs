async function getData() {
  const carsDataResponse = await fetch(
    "https://storage.googleapis.com/tfjs-tutorials/carsData.json"
  );
  const carsData = await carsDataResponse.json();
  const cleaned = carsData
    .map((car) => ({
      mpg: car.Miles_per_Gallon,
      horsepower: car.Horsepower,
    }))
    .filter((car) => car.mpg != null && car.horsepower != null);
  // Google에서 호스팅한 JSON 파일에서 'cars' 데이터 세트를 로드합니다.
  // 이 데이터 세트에는 주어진 각 자동차에 대한 다양한 특징이 많이 포함되어 있습니다.
  // 이 튜토리얼에서는 마력과 갤런당 마일에 대한 데이터만 추출하려고 합니다.
  // 목표는 숫자 1개, 마력을 가져와 숫자 1개, 갤런당 마일을 예측하도록 모델을 학습시키는 것입니다.

  return cleaned;
}

async function run() {
  const data = await getData();
  const values = data.map((d) => ({
    x: d.horsepower,
    y: d.mpg,
  }));

  tfvis.render.scatterplot(
    { name: "Horsepower v MPG" },
    { values },
    {
      xLabel: "Horsepower",
      yLabel: "MPG",
      height: 300,
    }
  );
  //이 패널을 바이저라고 하며 tfjs-vis에서 제공합니다. 이 패널에 시각화를 편리하게 표시할 수 있습니다.

  const model = createModel();
  tfvis.show.modelSummary({ name: "Model Summary" }, model);
  // 모델의 인스턴스가 생성되고 웹페이지에 레이어 요약이 표시됩니다.

  const tensorData = convertToTensor(data);
  const { inputs, labels } = tensorData;
  // Convert the data to a form we can use for training.

  await trainModel(model, inputs, labels);
  console.log("Done Training");
  // Train the model

  testModel(model, data, tensorData);
  // Make some predictions using the model and compare them to the
  // original data
}

function createModel() {
  const model = tf.sequential();
  // 입력이 출력으로 곧바로 흘러가므로 sequential 모델,
  // 다른 종류의 모델에는 분기 또는 여러 입력 및 출력이 있을 수 있지만 대부분의 경우 모델은 순차적.
  model.add(tf.layers.dense({ inputShape: [1], units: 1, useBias: true }));
  // dense 레이어는 행렬(가중치라고 함)에 입력을 곱한 후 그 결과에 숫자(편향이라고 함)를 더하는 레이어 유형입니다.
  // 네트워크의 첫 번째 레이어이므로 inputShape를 정의해야 합니다.
  // inputShape는 입력(특정 자동차의 마력)으로 숫자 1이 있으므로 [1]입니다.
  // units는 레이어에서 가중치 행렬의 크기를 설정합니다.
  // 1로 설정하면 데이터의 입력 특징별로 가중치가 1이 됩니다.
  model.add(tf.layers.dense({ units: 50 }));
  model.add(
    tf.layers.dense({ units: 1, activation: "sigmoid", useBias: true })
  );
  return model;
}

function convertToTensor(data) {
  return tf.tidy(() => {
    tf.util.shuffle(data);
    // 여기서는 학습 알고리즘에 제공할 예시의 순서를 무작위로 지정합니다.
    // 일반적으로 학습하는 동안 데이터 세트는 모델이 학습할 크기가 작은 하위 집합(배치라고 함)으로 분할되기 때문에 셔플이 중요합니다.
    // 셔플은 각 배치에 전체 데이터 분포의 데이터가 다양하게 포함되도록 하는 데 도움이 됩니다.
    // 데이터가 다양하게 포함되도록 하면 모델에 다음과 같은 이점이 있습니다.
    // 제공된 데이터의 순서에만 의존하여 학습하지 않도록 합니다.
    // 하위 그룹의 구조에 민감해지지 않도록 합니다.
    // 예를 들어 학습의 전반부에만 높은 마력의 자동차가 있다면 나머지 데이터 세트에는 적용되지 않는 관계를 학습할 수 있습니다.
    // 권장사항 : TensorFlow.js의 학습 알고리즘에 대해 데이터를 처리하기 전에 항상 데이터를 셔플링해야 합니다.

    const inputs = data.map((d) => d.horsepower);
    const labels = data.map((d) => d.mpg);
    const inputTensor = tf.tensor2d(inputs, [inputs.length, 1]);
    const labelTensor = tf.tensor2d(labels, [labels.length, 1]);
    // 여기서는 두 개의 배열을 만듭니다. 하나는 입력 예시(마력 항목)용이고 다른 하나는 실제 출력 값(머신러닝에서 라벨이라고 함)을 위한 배열입니다.
    // 그런 다음 각 배열 데이터를 2D 텐서로 변환합니다.
    // 텐서의 모양은 [num_examples, num_features_per_example]입니다. 여기에는 inputs.length 예시가 있으며 각 예시에는 1 입력 특징(마력)이 있습니다.

    const inputMax = inputTensor.max();
    const inputMin = inputTensor.min();
    const labelMax = labelTensor.max();
    const labelMin = labelTensor.min();
    const normalizedInputs = inputTensor
      .sub(inputMin)
      .div(inputMax.sub(inputMin));
    const normalizedLabels = labelTensor
      .sub(labelMin)
      .div(labelMax.sub(labelMin));
    // 다음으로 머신러닝 학습을 위한 또 다른 권장사항을 수행합니다.
    // Google에서는 데이터를 정규화합니다. 여기서는 최소-최대 조정을 사용하여 데이터를 숫자 범위 0-1로 정규화합니다.
    // Tensorflow.js로 빌드할 많은 머신러닝 모델의 내부는 너무 크지 않은 숫자에 대해 작동하도록 설계되기 때문에 정규화가 중요합니다.
    // 데이터를 정규화하는 일반적인 범위는 0 to 1 또는 -1 to 1입니다.
    // 데이터를 합당한 범위로 정규화하는 습관을 들이면 모델을 보다 성공적으로 학습시키게 됩니다.
    // 권장사항 : 학습 전에 항상 데이터 정규화를 고려해야 합니다. 정규화 없이 학습시킬 수 있는 데이터 세트도 있지만 데이터를 정규화하면 효과적인 학습을 방해하는 문제가 완전히 사라지는 경우가 종종 있습니다.
    // 데이터를 텐서로 변환하기 전에 정규화할 수 있습니다. 여기서는 TensorFlow.js의 벡터화를 활용해 루프를 위한 명시적 구문을 작성할 필요 없이 최소-최대 조정 작업을 수행할 수 있으므로 변환하기 전에 정규화합니다.

    return {
      inputs: normalizedInputs,
      labels: normalizedLabels,
      inputMax,
      inputMin,
      labelMax,
      labelMin,
    };
    // 출력을 정규화하지 않은 원래의 상태로 다시 되돌려서 원래의 조정으로 가져오고 향후 입력 데이터를 동일한 방식으로 정규화할 수 있도록 학습 중에 정규화에 사용한 값을 유지하려고 합니다.
  });
}

async function trainModel(model, inputs, labels) {
  model.compile({
    optimizer: tf.train.adam(),
    loss: tf.losses.meanSquaredError,
    metrics: ["mse"],
  });
  // 모델을 학습하기 전에 먼저 '컴파일'해야 합니다.
  // 그러려면 다음과 같이 매우 중요한 항목들을 지정해야 합니다.
  // optimizer: 모델이 예시를 보면서 업데이트하는 데 적용될 알고리즘입니다.
  // TensorFlow.js에는 다양한 옵티마이저가 있지만 여기에서는 실제로 매우 효과적이며 구성이 필요 없는 adam 옵티마이저를 선택했습니다.
  // loss: 표시되는 각 배치(데이터 하위 집합)를 얼마나 잘 학습하고 있는지 모델에 알려줄 함수입니다.
  // 여기서는 meanSquaredError를 사용해 모델이 수행한 예측을 실제 값과 비교합니다.

  const batchSize = 32;
  const epochs = 50;
  // 다음으로 batchSize와 세대의 수를 선택합니다.
  // batchSize는 각 학습 반복에서 모델이 보게 될 데이터 하위 집합의 크기를 나타냅니다.
  // 일반적인 배치 크기는 32~512 범위입니다. 모든 문제에 맞는 이상적인 배치 크기는 없으며 다양한 배치 크기의 수학적 연계성에 대한 설명은 이 튜토리얼에서 다루지 않습니다.
  // epochs는 모델이 제공된 전체 데이터 세트를 볼 횟수입니다. 여기서는 데이터 세트를 50번 반복합니다.

  return await model.fit(inputs, labels, {
    batchSize,
    epochs,
    shuffle: true,
    callbacks: tfvis.show.fitCallbacks(
      { name: "Training Performance" },
      ["loss", "mse"],
      { height: 200, callbacks: ["onEpochEnd"] }
    ),
  });
  // model.fit은 학습 루프를 시작하기 위해 호출하는 함수입니다.
  // 비동기 함수이므로 제공된 프라미스를 반환하여 학습이 완료되는 시기를 호출자가 결정할 수 있습니다.
  // 학습 진행 상황을 모니터링하기 위해 model.fit에 일부 콜백을 전달합니다.
  // tfvis.show.fitCallbacks를 사용해 앞에서 지정한 '손실' 및 'MSE' 측정항목에 대한 차트를 그리는 함수를 생성합니다.
}

function testModel(model, inputData, normalizationData) {
  const { inputMax, inputMin, labelMin, labelMax } = normalizationData;

  const [xs, preds] = tf.tidy(() => {
    const xs = tf.linspace(0, 1, 100);
    const preds = model.predict(xs.reshape([100, 1]));
    // 모델에 제공할 새 '예시'를 100개 생성합니다. Model.predict는 이러한 예시를 모델에 제공하는 방법입니다.
    // 학습할 때와 유사한 형태([num_examples, num_features_per_example])여야 합니다.

    const unNormXs = xs.mul(inputMax.sub(inputMin)).add(inputMin);
    const unNormPreds = preds.mul(labelMax.sub(labelMin)).add(labelMin);
    // 데이터를 0~1이 아닌 원래 범위로 되돌리려면 정규화 중에 계산한 값을 사용하고 연산을 반전시킵니다.

    return [unNormXs.dataSync(), unNormPreds.dataSync()];
  });
  // .dataSync()는 텐서에 저장된 값의 typedarray를 가져오는 데 사용할 수 있는 메서드입니다.
  // 이 메서드를 통해 일반 자바스크립트에서 해당 값을 처리할 수 있습니다. 일반적으로 권장되는 .data() 메서드의 동기식 버전입니다.

  const predictedPoints = Array.from(xs).map((val, i) => {
    return { x: val, y: preds[i] };
  });

  const originalPoints = inputData.map((d) => ({
    x: d.horsepower,
    y: d.mpg,
  }));

  tfvis.render.scatterplot(
    { name: "Model Predictions vs Original Data" },
    {
      values: [originalPoints, predictedPoints],
      series: ["original", "predicted"],
    },
    {
      xLabel: "Horsepower",
      yLabel: "MPG",
      height: 300,
    }
  );
  // 마지막으로 tfjs-vis를 사용해 원본 데이터와 모델의 예측을 표시합니다.
}

document.addEventListener("DOMContentLoaded", run);
