var express = require('express');
var http = require('http');
var path = require('path');
var brain = require('brain.js')

var app = express();
var net;
app.use(express.static("."));

function DayForRestuarant(restuarantName) {
    const result = net.run({ [restuarantName]: 1 });
    let highestValue = 0;
    let highestDay = '';
    for (let day in result) {
        if (result[day] > highestValue) {
            highestValue = result[day];
            highestDay = day;
        }
    }
    
    return highestDay;
}

//var request = require('request'); // "Request" library

const restaurants = {
    "Brilliant Yellow Corral": "Monday",
    "Penny’s": "Tuesday",
    "Right Coast Wings": "Wednesday",
    "The Delusion Last Railway Car": "Thursday",
    "Fun Day Inn": "Friday",
    "JHOP": "Saturday",
    "Owls": "Sunday"
};

//For the default site with no requests
app.get('/', function(req, res){
    res.sendFile(path.join(__dirname, 'index.html'));
});

app.get('/train', function(req, res){
    // input: { Monday, Tuesday, Wednesday, etc. }
    // output: { Restaurant1, Restaurant2 }

    const trainingData = [];

    for (let restaurantName in restaurants) {
        const dayOfWeek = restaurants[restaurantName];
        trainingData.push({
            input: { [restaurantName]: 1 },
            output: { [dayOfWeek]: 1 }
        });
    }

    net = new brain.NeuralNetwork({ hiddenLayers: [3] });

    const stats = net.train(trainingData);

    console.log(stats);
    res.send(stats);
});

app.get('/test', function(req, res){
    var results = DayForRestuarant("Right Coast Wings")
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
