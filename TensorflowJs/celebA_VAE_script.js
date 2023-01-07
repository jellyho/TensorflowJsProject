img = document.getElementById('upload');

function loadFile(input) {
    var file = input.files[0];	//선택된 파일 가져오기

    //이미지 source 가져오기
    img.src = URL.createObjectURL(file);
};