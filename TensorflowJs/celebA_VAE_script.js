img = document.getElementById('upload');
canvas = document.getElementById("constructed");

const encoder = tf.loadGraphModel("https://raw.githubusercontent.com/jellyho/TensorflowJsProject.github.io/master/TensorflowJs/celebA_VAE_encoder/model.json");
const decoder = tf.loadGraphModel("https://raw.githubusercontent.com/jellyho/TensorflowJsProject.github.io/master/TensorflowJs/celebA_VAE_decoder/model.json");

var latent = tf.zeros([1, 400]);
var constructed;
var trait_vectors;

function loadFile(input) {
    var file = input.files[0];	//선택된 파일 가져오기

    //이미지 source 가져오기
    img.src = URL.createObjectURL(file);
    encode();
    decode();
};

traits = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair'
    , 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male'
    , 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns'
    , 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young'];

train_values = {}
sliders = {}
slider_ps = {}

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

resetTraits();
fetch('https://raw.githubusercontent.com/jellyho/TensorflowJsProject.github.io/master/TensorflowJs/trait_vectors.json').then((response) => response.json()).then((json) => trait_vectors=json);

function update_traits() {
    var new_latent = tf.clone(latent[0]);
    for (let i = 0; i < traits.length; i++) {
        var trait = tf.tensor(trait_vectors[traits[i]]);
        trait = trait.mul(train_values[traits[i]]);
        trait = tf.reshape(trait, [1, 400]);      
        new_latent = new_latent.add(trait);
    }
    console.log(new_latent.print());
    decode_new(new_latent);
}

function encode()
{
    var input = tf.image.resizeBilinear(tf.browser.fromPixels(img), [128, 128]);
    input = tf.reshape(input, [1, 128, 128, 3]);
    input = tf.div(input, 255.0);
    encoder.then(function (encoder) {latent = encoder.predict(input);});
    console.log(latent);
}

function decode()
{
    var input = tf.reshape(latent[0], [1, 400]);
    decoder.then(function (decoder) {constructed = decoder.predict(input);});
    constructed = tf.reshape(constructed, [128, 128, 3]);
    tf.browser.toPixels(constructed, canvas);
}

function decode_new(latent) {
    var input = tf.reshape(latent, [1, 400]);
    decoder.then(function (decoder) { constructed = decoder.predict(input); });
    constructed = tf.reshape(constructed, [128, 128, 3]);
    tf.browser.toPixels(constructed, canvas);
}