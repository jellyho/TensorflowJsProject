img = document.getElementById('upload');
canvas = document.getElementById("constructed");
state = document.getElementById("state");   

const encoder = tf.loadGraphModel("https://raw.githubusercontent.com/jellyho/TensorflowJsProject.github.io/master/TensorflowJs/celebA_VAE_encoder/model.json");
const decoder = tf.loadGraphModel("https://raw.githubusercontent.com/jellyho/TensorflowJsProject.github.io/master/TensorflowJs/celebA_VAE_decoder/model.json");

var latent = tf.zeros([1, 400]);
var constructed;
var trait_vectors;

function loadFile(input) {
    var file = input.files[0];	//선택된 파일 가져오기

    //이미지 source 가져오기
    img.src = URL.createObjectURL(file);
    console.log(img.src);
    console.log('image loaded');
    setTimeout(function () { encode();}, 1000);
    setTimeout(function () { decode();}, 2000);
};

function setExampleImage(obj) {
    console.log('image loaded');
    img.src = obj.src;
    console.log(img.src);
    resetBars();
    setTimeout(function () { encode(); }, 1000);
    setTimeout(function () { decode(); }, 2000);
}

traits = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair'
    , 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male'
    , 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns'
    , 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young'];

train_values = {}
sliders = {}
slider_ps = {}

function resetBars() {
    for (let i = 0; i < traits.length; i++) {
        train_values[traits[i]] = 0;
        slider_ps[traits[i]].innerText = 0;
    }
}

function resetTraits() {
    for (let i = 0; i < traits.length; i++) {
        train_values[traits[i]] = 0;
        var slider = document.getElementById(traits[i]);
        sliders[traits[i]] = slider;
        var sliderp = document.getElementById('vec_' + traits[i]);
        slider_ps[traits[i]] = sliderp;
        slider.addEventListener('input', (event) => {
            var value = (event.target.value - 50) / 10;
            slider_ps[event.target.id].innerText = value;
            train_values[event.target.id] = value;
            update_traits();
        });
    }
    return true;
}


function update_traits() {
    var new_latent = tf.clone(latent[0]);
    for (let i = 0; i < traits.length; i++) {
        var trait = tf.tensor(trait_vectors[traits[i]]);
        trait = trait.mul(train_values[traits[i]]);
        trait = tf.reshape(trait, [1, 400]);
        new_latent = new_latent.add(trait);
    }
    var d = latent[0].norm();
    new_latent = new_latent.div(new_latent.norm()).mul(d);
    decode_new(new_latent);
}

function encode() {
    state.innerText = "encoding...";
    state.style = "width : 25%";
    var input = tf.image.resizeBilinear(tf.browser.fromPixels(img), [128, 128]);
    input = tf.reshape(input, [1, 128, 128, 3]);
    input = tf.div(input, 255.0);
    encoder.then(function (encoder) {
        latent = encoder.predict(input);
        console.log(latent[0].print());
    });
    state.innerText = "encoded";
    state.style = "width : 50%";
}

function decode() {
    state.innerText = "decoding...";
    state.style = "width : 75%";
    var input = tf.reshape(latent[0], [1, 400]);
    decoder.then(function (decoder) {
        constructed = decoder.predict(tf.reshape(input, [1, 400]));
        constructed = tf.reshape(constructed, [128, 128, 3]);
        tf.browser.toPixels(constructed, canvas);
        tf.dispose(constructed);
    });
    state.innerText = "decoded";
    state.style = "width : 100%";
}

function decode_new(latent) {
    var input = tf.reshape(latent, [1, 400]);
    decoder.then(function (decoder) { constructed = decoder.predict(input); });
    constructed = tf.reshape(constructed, [128, 128, 3]);
    tf.browser.toPixels(constructed, canvas);
    tf.dispose(constructed);
}

resetTraits();
fetch('https://raw.githubusercontent.com/jellyho/TensorflowJsProject.github.io/master/TensorflowJs/trait_vectors.json').then((response) => response.json()).then((json) => trait_vectors=json);
