var express = require('express');
var fs = require('fs')
var app = express()

app.use(express.bodyParser())
app.configure(function() {
  app.use('/files', express.static(__dirname + '/files'));
})
app.listen(3333, "localhost", function() {
  console.log('Server Start');
})

app.get('/', function(req, res) {
  fs.readFile('index.html',function (error, data) {
    res.writeHead(200, { "Content-type":'text/html'})
    res.end(data, function(error) {
      console.log(error)
    });
  })
})

app.post('/upload',function( req, res) {
  console.log(req);
  fs.readFile(req.files.uploadFile.path, function(error, data) {
    var filePath = __dirname+"/files/"+req.files.uploadFile.name;
    fs.writeFile(filePath, data, function(error) {
      if (error) {
      } else {
        res.redirect("back");
      }
    });
  });
});
