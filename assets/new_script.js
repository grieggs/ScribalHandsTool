let canvas = this.__canvas = new fabric.Canvas('c');
// let canvas2 = this.__canvas = new fabric.Canvas('c2');
let image_group = new fabric.Group([]);


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
      var score = document.getElementById("score")
      e.target.set('stroke', 'red');
      e.target.set('strokeWidth', 3);
      e.target.set('left', e.target.get('left')-1.5);
      e.target.set('top', e.target.get('top')-1.5);
      focus_img.src = e.target.toDataURL();
      score.innerText = Math.floor(Math.random()*100)
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
    canvas.sendToBack(e.target);
    canvas.renderAll();
  }
});


function loadImage() {
      // var canvas = document.getElementById("c").fabric;
    myPromise = window.pywebview.api.loadImage()
    myPromise.then(
        function(value) {
            var i = 0;
            var j = 0;
            image_group = new fabric.Group([])

            value = JSON.parse(value)
            for (let i = 0, len = value.length; i < len; i++) {
                for (let j = 0, len = value[i].length; j < len; j++) {
                   fabric.Image.fromURL(value[i][j], function(myImg) {
                         //i create an extra var for to change some image properties
                         var img1 = myImg.set({ left: j*64, top: i*64 ,width:64,height:64});
                         img1.selectable = false;
                         image_group.add(img1)
                         canvas.add(img1);
                    });
                }
            }


        },
        function(error) { /* code if some error */ }
        );

    }
function loadTemplate() {

}








