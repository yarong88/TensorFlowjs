const { Server } = require("socket.io");
const http = require("http");
const express = require("express");
const path = require("path");
const logger = require("morgan");
const multer = require("multer");
const app = express();

const server = http.createServer(app);
const io = new Server(server);

const port = 3000;
const _path = path.join(__dirname, "/");
console.log(_path);
app.use("/", express.static(_path));
app.use(logger("tiny"));

app.use(express.json());
app.use(
  express.urlencoded({
    extended: true,
  })
);

require("dotenv").config();

let client_id = process.env.papago_id;
let client_secret = process.env.papago_secret;

let api_url = "https://openapi.naver.com/v1/papago/n2mt";
let request = require("request");
io.on("connection", (socket) => {
  socket.on("translate", (msg) => {
    let options = {
      url: api_url,
      form: { source: "en", target: "ko", text: msg },
      headers: {
        "X-Naver-Client-Id": client_id,
        "X-Naver-Client-Secret": client_secret,
      },
    };
    request.post(options, function (error, response, body) {
      if (!error) {
        const rst = JSON.parse(body);
        io.emit("translate", rst.message.result.translatedText);
      } else {
        console.log("error = " + response.statusCode);
      }
    });
  });
});

const storage = multer.diskStorage({
  destination: (req, res, cb) => {
    cb(null, _path);
  },
  filename: (req, res, cb) => {
    cb(null, "dog.jpg");
  },
});

let upload = multer({ storage: storage });

app.post("/up", upload.single("ufile"), (req, res) => {
  console.log(req.file);
  res.send(
    `<script>alert("파일 업로드 완료");location.replace("index.html")</script>`
  );
});

server.listen(port, function () {
  console.log("translate app listening on port!");
});
