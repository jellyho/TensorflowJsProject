// JavaScript source code
const model = tf.loadGraphModel("https://raw.githubusercontent.com/jellyho/TensorflowJs/master/TensorflowJs/MNIST_VAE_DECODER/model.json");

context_canvas = document.getElementById("context");
number_canvas = document.getElementById("number");
context_pos_text = document.getElementById("latent_pos")
context_ctx = context_canvas.getContext("2d");
number_ctx = number_canvas.getContext("2d");


number_ctx.fillStyle = "black";
number_ctx.fillRect(0, 0, 400, 400);

latent_img = new Image();
latent_img.src = "mnist_latent_croped.png";

latent_img.onload = function () //이미지 로딩 완료시 실행되는 함수
{
    context_ctx.drawImage(latent_img, 0, 0, 400, 400);
}

var latent_clicked = false;
var latent_x = 0.5;
var latent_y = 0.5;
var event_clientX = 0;
var event_clientY = 0;
var input;
var pred;

function erfc(x) {
    var z = Math.abs(x);
    var t = 1 / (1 + z / 2);
    var r = t * Math.exp(-z * z - 1.26551223 + t * (1.00002368 +
            t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 +
            t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 +
            t * (-0.82215223 + t * 0.17087277)))))))))
    return x >= 0 ? r : 2 - r;
  };

function ierfc(x)
{
    if (x >= 2) {return -100;}
    if (x <= 0) {return 100;}

    var xx = (x < 1) ? x : 2 - x;
    var t = Math.sqrt(-2 * Math.log(xx / 2));

    var r = -0.70711 * ((2.30753 + t * 0.27061) /
            (1 + t * (0.99229 + t * 0.04481)) - t);

    for (var j = 0; j < 2; j++) {
      var err = erfc(r) - xx;
      r += err / (1.12837916709551257 * Math.exp(-(r * r)) - r * err);
    }

    return (x < 1) ? r : -r;
}

function ppf(x)
{
    return -Math.sqrt(2) * ierfc(2 * x);
}

context_canvas.addEventListener('mousedown', function (e) {
    latent_clicked = true;
    event_clientX = e.clientX;
    event_clientY = e.clientY;
});
context_canvas.addEventListener('mousemove', function (e) {
    if (latent_clicked) {
        event_clientX = e.clientX;
        event_clientY = e.clientY;

        const rect = context_canvas.getBoundingClientRect();
        context_ctx.drawImage(latent_img, 0, 0, 400, 400);
        latent_x = event_clientX - rect.left;
        latent_y = event_clientY - rect.top;
        context_ctx.fillStyle = "black";
        context_ctx.beginPath();
        context_ctx.arc(latent_x, latent_y, 5, 0, Math.PI * 2)
        context_ctx.fill();
        latent_y = 400 - latent_y;
        latent_x /= 400;
        latent_y /= 400;
        var input_x = ppf(latent_x);
        var input_y = ppf(latent_y);
        if (model)
        {
            input = tf.tensor([[input_x, input_y]]);
            model.then(function (model) {
                pred = model.predict(input);
            });
            if(pred)
            {
                pred = tf.reshape(pred, [28, 28]);
                tf.browser.toPixels(pred, number_canvas);
            }
        }
        context_pos_text.innerText = "[ " + latent_x + ", " + latent_y + " ]";
    }
});
context_canvas.addEventListener('mouseup', function (e) {
    latent_clicked = false;
    event_clientX = 0;
    event_clientY = 0;
    context_pos_text.innerText = "[ " + latent_x + ", " + latent_y + " ]";
});