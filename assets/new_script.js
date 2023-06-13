

function fun()
{
    myPromise = window.pywebview.api.loadImage()
    myPromise.then(
        function(value) {
            for (let i = 0, len = value.length, text = ""; i < len; i++) {
                for (let j = 0, len = value[i].length, text = ""; i < len; i++) {
                    var img = document.createElement("img");
                    img.src = value[i][j];
                    img.className = "center-fit"
                    var src = document.getElementById("image");
                    src.appendChild(img);
                }
            }
            // document.getElementById("result2").innerHTML = value;
        },
        function(error) { /* code if some error */ }
        );

}