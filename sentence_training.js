var express = require('express');
var http = require('http');
var path = require('path');
var brain = require('brain.js')

var app = express();
var net;
app.use(express.static("."));

function result(phrase){
    let result = net.run(phrase);
    return result;
}
/*
const trainingData = [
  'Jane saw Doug.',
  'Doug saw Jane.',
  'Spot saw Doug and Jane looking at each other.',
  'It was love at first sight, and Spot had a frontrow seat. It was a very special moment for all.'
];
*/

var fs = require('fs');
var textByLine = fs.readFileSync('jp.txt').toString().split("\n");
console.log(textByLine)
let trainingData = textByLine
//For the default site with no requests
app.get('/', function(req, res){
    res.sendFile(path.join(__dirname, 'index.html'));
});

app.get('/train', function(req, res){
    console.log('Training');
    net = new brain.recurrent.LSTM();
    const stats = net.train(trainingData, {
        iterations: 1500,
        errorThresh: 0.011
    });
    console.log(stats);
    res.send(stats);
});

app.get('/test', function(req, res){
    console.log('Testing');
    var results = result('love')
    res.send(results);
});

/*
console.log(DayForRestuarant('Brilliant Yellow Corral'));
console.log(DayForRestuarant('Penny’s'));
console.log(DayForRestuarant('Right Coast Wings'));
console.log(DayForRestuarant('The Delusion Last Railway Car'));
console.log(DayForRestuarant('Fun Day Inn'));
console.log(DayForRestuarant('JHOP'));
console.log(DayForRestuarant('Owls'));
*/
app.listen(8080,function(){
    console.log('Server is listening :]');
});
