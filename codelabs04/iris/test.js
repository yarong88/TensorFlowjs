const iris = require("./iris.json");
const IRIS_DATA = [];
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
console.log(IRIS_DATA);
