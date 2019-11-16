let inputValues = 9;
let testValue = [1,0,1,1,1,1,1,1,1];

async function run() {
    document.getElementById("output_field").innerText = "...training...";
    const data = await getData();
    const tensorData = prepareData(data);
    const {inputs,outputs} = tensorData;
    train(inputs, outputs);
}

async function getData() {
    const dataReq = await fetch("./data.json");
    const data = dataReq.json();
    return data;
}

function prepareData(data) {
    return tf.tidy(() => {
        tf.util.shuffle(data);

        const inputs = data.map(d => [d.q1, d.q2, d.q3, d.q4, d.q5, d.q6, d.q7, d.q8, d.q9]);
        const outputs = data.map(d => d.age);

        return {
            inputs: inputs,
            outputs: outputs,
        }
    });
}

//train
async function train(inputs,outputs) {
    const model = tf.sequential();
    model.add(tf.layers.dense({units: 1, inputShape:[9]}));

    model.compile({
        loss: 'meanSquaredError',
        optimizer: 'sgd'
    });

    const xs = tf.tensor2d(inputs,[inputs.length,inputValues]);
    const ys = tf.tensor2d(outputs,[outputs.length,1]);

    await model.fit(xs,ys,{epochs:500});

    let cards = document.getElementsByClassName("card");
    for (var i = 0; i < cards.length; i++) {
        testValue[i] = cards[i].classList.contains("active") ? 1 : 0;
    }

    document.getElementById("output_field").innerText = model.predict(tf.tensor2d(testValue,[1,inputValues]));
}


function showQuiz() {
    document.getElementById("quiz").style.display = "block";
}

document.getElementById("quiz").style.display = "none";

function activate(_el) {
    if (_el.classList.contains("active")) {
        _el.classList.remove("active");
    }
    else {
        _el.classList.add("active");
    }
}