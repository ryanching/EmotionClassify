var express = require("express");
var app = require('express')();
var http = require('http').Server(app);
var io = require('socket.io')(http);
var sd = require('standard-deviation')
var svm = require('node-svm');
var fs = require('fs');

//Make examples and build directories accessible for index.html
app.use("/examples", express.static(__dirname + '/examples'));
app.use("/build", express.static(__dirname + '/build'));
app.use("/js", express.static(__dirname + '/js'));
app.use("/media", express.static(__dirname + '/media'));

//Listen on port 3000
http.listen(3000, function () {
    console.log('listening on *:3000');
});

//Serve index.html
app.get('/', function(req, res){
    res.sendFile(__dirname + '/index.html');
});

//Load CK+ dataset coordinates and labels to train svm on
var svm_input_train, featurePoints, svm_input_test;
svm_input_train = fs.readFileSync('./clm_traindata4node.txt','utf8');
svm_input_train = JSON.parse(svm_input_train)
svm_input_test = fs.readFileSync('./clm_valdata4node.txt','utf8');
svm_input_test = JSON.parse(svm_input_test);

// test model on 80% train, 20% val data
var clf_test = new svm.CSVC({probability: true, kernelType: 'POLY', degree:[3], gamma:[0.5], normalize:false});
clf_test.train(svm_input_train).done(function () {
    eval = clf_test.evaluate(svm_input_test);
});

// train model on full dataset for use with live video
full_data = svm_input_train.slice(0); // copy array
svm_input_test.forEach(function(ex){
    full_data.push(ex);
})

var clf = new svm.CSVC({probability: true, kernelType: 'POLY', degree:[3], gamma:[0.5], normalize:false});
clf.train(full_data).done(function () {
});

//Listen for user connection (open localhost:3000)
io.on('connection', function(socket){
  console.log('a user connected');
  socket.on('disconnect', function(){
    console.log('user disconnected');
  });
  socket.on('init', function(msg){
      console.log('message: ' + msg);
    });
    socket.on('featurePoints', function(msg){
    featurePoints = msg;
    //Take passed featurepoints from client webcam and perform procrustes allignment
    if(featurePoints != null) {
        var result = procrustes(avgNeutral, featurePoints);
        var procrustesParams = [result[0], result[1], result[2], result[3]];
        var curFace = applyProcrustes(featurePoints, procrustesParams);

        //calculate difference between webcam face and average neutral of dataset
        var diff = []
        for (var i = 0; i < curFace.length; i++) {
           diff[i] = avgNeutral[i][0] - curFace[i][0] ; //diff x
           diff[i + curFace.length] = avgNeutral[i][1] - curFace[i][1]; //diff y
        }

        //normalize the difference data
       var sum = 0;
       var std = sd(diff);
       for (var i = 0; i < diff.length; i++) {
           sum += diff[i]/diff.length;
       }
       for (var i = 0; i < diff.length; i++) {
           diff[i] = diff[i]/std;
           diff[i] = diff[i] - sum;
       }
       //classify the emotion expressed in webcam
        prob = clf.predictProbabilitiesSync(diff);
    };

    //Return the classified emotion probabilities to be displayed
    io.emit('featurePointsResponse', prob);

    });

});



//Applies procrustesParams to face parameter and returns aligned face coordinates
function applyProcrustes(face, procrustesParams) {
    var finalFace = face;

    translateX = procrustesParams[0];
    translateY = procrustesParams[1];
    scale = procrustesParams[2];
    rotate = procrustesParams[3];

    for (var i = 0; i < face.length; i++) {
        x = finalFace[i][0];
        y = finalFace[i][1];
        a = ((scale * Math.cos(rotate)) - 1) * x - (scale * Math.sin(rotate)) * y + translateX;
        b = ((scale * Math.cos(rotate)) - 1) * y + (scale * Math.sin(rotate)) * x + translateY;
        finalFace[i][0] += a;
        finalFace[i][1] += b;
    }

    return finalFace;
}

//Performs procrustes allignment from shape to template
function procrustes(template, shape) {
    // assume template and shape is a vector of x,y-coordinates
    //i.e. template = [[x1,y1], [x2,y2], [x3,y3]];
    var templateClone = [];
    var shapeClone = [];
    for (var i = 0; i < template.length; i++) {
        templateClone[i] = [template[i][0], template[i][1]];
    }
    for (var i = 0; i < shape.length; i++) {
        shapeClone[i] = [shape[i][0], shape[i][1]];
    }
    shape = shapeClone;
    template = templateClone;

    // calculate translation
    var templateMean = [0.0, 0.0];
    for (var i = 0; i < template.length; i++) {
        templateMean[0] += template[i][0];
        templateMean[1] += template[i][1];
    }
    templateMean[0] /= template.length;
    templateMean[1] /= template.length;

    var shapeMean = [0.0, 0.0];
    for (var i = 0; i < shape.length; i++) {
        shapeMean[0] += shape[i][0];
        shapeMean[1] += shape[i][1];
    }
    shapeMean[0] /= shape.length;
    shapeMean[1] /= shape.length;

    var translationX = templateMean[0] - shapeMean[0];
    var translationY = templateMean[1] - shapeMean[1];

    // centralize
    for (var i = 0; i < shape.length; i++) {
        shape[i][0] -= shapeMean[0];
        shape[i][1] -= shapeMean[1];
    }
    for (var i = 0; i < template.length; i++) {
        template[i][0] -= templateMean[0];
        template[i][1] -= templateMean[1];
    }

    //Scales shape to match template
    var scaleS = 0.0;
    for (var i = 0; i < shape.length; i++) {
        scaleS += ((shape[i][0]) * (shape[i][0]));
        scaleS += ((shape[i][1]) * (shape[i][1]));
    }
    scaleS = Math.sqrt(scaleS / shape.length);

    var scaleT = 0.0;
    for (var i = 0; i < template.length; i++) {
        scaleT += ((template[i][0]) * (template[i][0]));
        scaleT += ((template[i][1]) * (template[i][1]));
    }
    scaleT = Math.sqrt(scaleT / template.length);

    var scaling = scaleT / scaleS;

    for (var i = 0; i < shape.length; i++) {
        shape[i][0] *= scaling;
        shape[i][1] *= scaling;
    }

    // rotation
    var top = 0.0;
    var bottom = 0.0;
    for (var i = 0; i < shape.length; i++) {
        top += (shape[i][0] * template[i][1] - shape[i][1] * template[i][0]);
        bottom += (shape[i][0] * template[i][0] + shape[i][1] * template[i][1]);
    }
    var rotation = Math.atan(top / bottom);

    translationX += (shapeMean[0] - (scaling * Math.cos(-rotation) * shapeMean[0]) - (scaling * shapeMean[1] * Math.sin(-rotation)));
    translationY += (shapeMean[1] + (scaling * Math.sin(-rotation) * shapeMean[0]) - (scaling * shapeMean[1] * Math.cos(-rotation)));

    return [translationX, translationY, scaling, rotation];
}

//Average  neutral face across the dataset
var avgNeutral = [
    [232.527014381771, 206.847371009327],
    [230.191059226508, 253.679385807828],
    [237.210969100666, 299.506491121061],
    [250.095631256741, 345.064473582933],
    [271.860958504404, 383.147139324297],
    [301.944126630250, 412.757957471166],
    [337.820387566316, 435.240163904126],
    [378.595415235547, 442.588526941324],
    [419.032690606763, 433.113015982257],
    [453.809965467365, 408.792251458362],
    [482.092694115350, 377.438094632230],
    [501.556066081682, 338.005258549214],
    [511.696663276935, 291.547457670306],
    [515.981119944862, 245.441615749393],
    [511.068775132303, 198.998831243039],
    [480.450978752522, 172.322849276516],
    [458.845057037992, 161.792805246925],
    [426.300418442928, 164.087517868959],
    [400.922746150755, 171.181430470465],
    [260.630004262959, 177.653492685632],
    [281.271648717855, 166.178402082895],
    [313.710549484431, 166.807934313647],
    [339.357281129112, 172.616176403508],
    [280.478069231013, 210.524986827478],
    [305.267341764553, 198.002627341728],
    [333.163069485981, 209.516920117069],
    [305.935918601799, 217.230585596006],
    [306.376269045302, 207.007721627402],
    [462.131162513540, 205.927300798791],
    [436.367928261444, 194.746655075876],
    [409.326604683205, 207.666757110755],
    [436.785919529252, 213.940790031660],
    [435.592363981815, 203.791252179607],
    [371.003295663686, 198.388447508914],
    [343.557665459225, 258.399679691668],
    [331.925479232997, 279.342787973510],
    [343.419611757210, 293.362770804058],
    [373.949328753945, 297.668751110672],
    [404.245540524710, 291.784410086814],
    [415.039617230313, 277.067212034114],
    [402.202124558237, 256.807238369720],
    [372.156394651273, 237.306319248579],
    [352.753875947565, 285.268035179832],
    [394.495732527321, 284.166474363470],
    [322.755748321485, 343.782283107925],
    [340.884304148450, 330.821900692369],
    [361.132772924126, 325.300045129754],
    [374.856030392684, 327.886980629704],
    [388.471522732578, 324.569719009213],
    [409.095257320502, 329.046014077346],
    [428.086585603721, 341.041156224660],
    [416.076576821152, 358.357063864207],
    [399.303211991842, 369.763644011573],
    [376.216272394510, 373.604238062969],
    [352.940267771509, 370.911476982738],
    [335.735279269901, 360.428757040150],
    [349.694863316080, 351.528894264967],
    [375.699521774823, 354.453283217651],
    [401.504618649747, 350.242481711109],
    [401.114070681496, 337.414208784794],
    [375.219243120761, 338.291808389582],
    [349.259853895394, 338.776670996389],
    [373.216276719997, 274.184811223969],
    [290.659538250067, 201.892656366791],
    [321.367232574625, 200.782584553876],
    [320.271218920617, 214.309717342772],
    [291.968293475925, 215.377425355955],
    [451.265289027457, 197.868138334319],
    [420.514152580511, 198.335253513266],
    [422.357078101686, 211.760156380024],
    [450.742847366686, 211.393693721650]];
