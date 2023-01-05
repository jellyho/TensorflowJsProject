// JavaScript source code
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
        latent_x = event_clientX - rect.left;
        latent_y = event_clientY - rect.top;
        latent_y = 400 - latent_y;
        latent_x /= 400;
        latent_y /= 400;
        context_pos_text.innerText = "[ " + latent_x + ", " + latent_y + " ]";
    }
});
context_canvas.addEventListener('mouseup', function (e) {
    latent_clicked = false;
    event_clientX = 0;
    event_clientY = 0;
    context_pos_text.innerText = "[ " + latent_x + ", " + latent_y + " ]";
});