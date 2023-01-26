fetch("./보스턴집값.json")
  .then((response) => {
    return response.json();
  })
  .then((jsondata) => {
    console.log(jsondata);
    // const database = [];
    // const dataresult = [];
    // for (let i = 0; i < jsondata.Data.length; i++) {
    //   const dataunit = [];
    //   dataunit.push(jsondata.Data[i].CRIM);
    //   dataunit.push(jsondata.Data[i].ZN);
    //   dataunit.push(jsondata.Data[i].INDUS);
    //   dataunit.push(jsondata.Data[i].CHAS);
    //   dataunit.push(jsondata.Data[i].NOX);
    //   dataunit.push(jsondata.Data[i].RM);
    //   dataunit.push(jsondata.Data[i].AGE);
    //   dataunit.push(jsondata.Data[i].DIS);
    //   dataunit.push(jsondata.Data[i].RAD);
    //   dataunit.push(jsondata.Data[i].TAX);
    //   dataunit.push(jsondata.Data[i].PTRATIO);
    //   dataunit.push(jsondata.Data[i].B);
    //   dataunit.push(jsondata.Data[i].LSTAT);
    //   database.push(dataunit);
    //   dataresult.push([jsondata.Data[i].MEDV]);
    // }
    // let xt = tf.tensor(database);
    // let yt = tf.tensor(dataresult);
    // model.fit(xt, yt, fitParm).then((_) => {
    //   let result = model.predict(xt);
    //   result.print();
    //   model.save("localstorage://my-model-1");
    //   console.log("모델 저장됨");
    // });
  });
