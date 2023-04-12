const tf = require('@tensorflow/tfjs-node');

async function run() {
    var loadedModel = await tf.loadGraphModel('file://./saved_js_models/my_model_1/model.json');

    // Predict a new value
    console.log(' \n ============= Predicted result using Tensorflow.js Model: =============');
    loadedModel.predict(tf.tensor2d([[9.85, 6900, 0, 6, 0, 1, 0, 1]])).print();

}
run();