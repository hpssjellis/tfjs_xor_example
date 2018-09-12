var tf = require('@tensorflow/tfjs');
require('@tensorflow/tfjs-node')

async function go() {
 
console.log('tf.version.tfjs: '+tf.version.tfjs)
console.log('tf.version_core: '+tf.version_core)
console.log('tf.version_layers: '+tf.version_layers)


const model = tf.sequential();
model.add(tf.layers.dense({units: 10, activation: 'sigmoid',inputShape: [2]}));
model.add(tf.layers.dense({units: 1, activation: 'sigmoid',inputShape: [10]}));

model.compile({loss: 'meanSquaredError', optimizer: 'rmsprop'});

const training_data = tf.tensor2d([[0,0],[0,1],[1,0],[1,1]]);
const target_data = tf.tensor2d([[0],[1],[1],[0]]);

for (let i = 1; i < 10 ; ++i) {
 var h = await model.fit(training_data, target_data, {epochs: 30});
   console.log("Loss after Epoch " + i + " : " + h.history.loss[0]);
}

   model.predict(training_data).print();

   await model.save('file:///home/ubuntu/workspace/my-model-4');   // why can't I name the files???
   const model2= await tf.loadModel('file:///home/ubuntu/workspace/my-model-4/model.json');   // here I can name the file
   await model2.summary()
   
   model2.predict(training_data).print();
}

go();
