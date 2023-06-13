let canvas = this.__canvas = new fabric.Canvas('c');
// let canvas2 = this.__canvas = new fabric.Canvas('c2');
let image_group = new fabric.Group([]);
let color_group = new fabric.Group([]);
let img_array = []
let color_array = []
let img_opacity = .5


window.addEventListener('load', function() {
    resizeCanvas()
})

function resizeCanvas(){
    const outerCanvasContainer = document.getElementById('fabric-canvas-wrapper');

    const ratio          = canvas.getWidth() / canvas.getHeight();
    const containerWidth = outerCanvasContainer.clientWidth;
    const scale          = containerWidth / canvas.getWidth();
    const zoom           = 1

    canvas.setDimensions({width: containerWidth, height: containerWidth / ratio});
    canvas.setViewportTransform([zoom, 0, 0, zoom, 0, 0]);
}
window.addEventListener('resize', resizeCanvas, true);



// document.getElementById('c').fabric = canvas;
fabric.Object.prototype.transparentCorners = false;
canvas.on('mouse:over', function(e) {
  if(e.target) {
      var focus_img = document.getElementById('focus');
      e.target.set('opacity',1)
      focus_img.src = e.target.toDataURL();
      e.target.set('opacity',img_opacity)
      var score = document.getElementById("score")
      score.innerText = e.target.score
      e.target.set('stroke', 'red');
      e.target.set('strokeWidth', 3);
      e.target.set('left', e.target.get('left')-1.5);
      e.target.set('top', e.target.get('top')-1.5);
      canvas.bringToFront(e.target);
      canvas.renderAll();
  }
});
canvas.on('mouse:out', function(e) {
  if(e.target){
    e.target.set('stroke', 'none');
    e.target.set('strokeWidth',0);
    e.target.set('left', e.target.get('left')+1.5);
    e.target.set('top', e.target.get('top')+1.5);
    // canvas2.clear();
    // canvas.sendBackwards(e.target);
    canvas.renderAll();
  }
});


function loadImage() {
    // var canvas = document.getElementById("c").fabric;
    myPromise = window.pywebview.api.loadImage()
    myPromise.then(
        function(value) {
            image_group = new fabric.Group([]);
            value = JSON.parse(value);
            img_array = [];
            for (let i = 0, len = value.length; i < len; i++) {
                img_array.push([]);
                for (let j = 0, len = value[i].length; j < len; j++) {
                    fabric.Image.fromURL(value[i][j], function (myImg) {
                        //i create an extra var for to change some image properties
                        var img1 = myImg.set({left: j * 64, top: i * 64, width: 64, height: 64});
                        img1.selectable = false;
                        img1.score = 'n/a';
                        image_group.add(img1)
                        canvas.add(img1);
                        img_array[i].push(img1)
                    });
                }
            }

        },
        function(error) { /* code if some error */ }
        );

};

function evalImage(){
    myPromise = window.pywebview.api.evalImage()
    myPromise.then(
        function(value) {
            color_array = []
            value = JSON.parse(value);
            for (let i = 0, len = value.length; i < len; i++) {
                color_array.push([])
                for (let j = 0, len = value[i].length; j < len; j++) {
                    img_array[i][j].score = value[i][j].toFixed(5)
                    img_array[i][j].set('opacity', img_opacity);
                    if(value[i][j] > .98){
                        var rect = new fabric.Rect({
                          left: j*64,
                          top: i*64,
                          fill: 'green',
                          width: 64,
                          height: 64
                        });
                    }else if (value[i][j] > .93) {
                        var rect = new fabric.Rect({
                          left: j*64,
                          top: i*64,
                          fill: 'yellow',
                          width: 64,
                          height: 64
                        });
                    } else {
                        var rect = new fabric.Rect({
                          left: j*64,
                          top: i*64,
                          fill: 'red',
                          width: 64,
                          height: 64
                        });
                    }
                    canvas.sendToBack(rect)
                    color_array[i].push(rect)
                    color_group.add(rect)
                }
            }
        },
        function(error) { /* code if some error */ }
        );
};

function toggleColors(){
    if(img_opacity == .5){
        img_opacity = 1;
        for (let i = 0, len = img_array.length; i < len; i++) {
            for (let j = 0, len = img_array[i].length; j < len; j++) {
                img_array[i][j].set('opacity', img_opacity);
            }
        }
    }else{
        img_opacity = .5;
        for (let i = 0, len = img_array.length; i < len; i++) {
            for (let j = 0, len = img_array[i].length; j < len; j++) {
                img_array[i][j].set('opacity', img_opacity);
            }
        }
    }
}



function loadTemplate() {

}








