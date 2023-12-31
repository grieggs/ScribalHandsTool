let canvas = this.__canvas = new fabric.Canvas('c');
// let canvas2 = this.__canvas = new fabric.Canvas('c2');
let image_group = new fabric.Group([]);
let color_group = new fabric.Group([]);
let img_array = [];
let color_array = [];
let img_opacity = 1;
let raw_width = -1;
let raw_height = -1;
let frozen = false;
let zoom = 1;
let avg_score = -1;

document.addEventListener('DOMContentLoaded', function() {
    setTimeout(function () {
        myPromise = window.pywebview.api.getTemplatePath();
        myPromise.then(
            function (value) {
                document.getElementById('template').innerHTML = value;
            },
            function (error) {
            });
    }, 100);
});



// window.addEventListener('load', function() {
//     resizeCanvas()
// })

// function resizeCanvas(){
//     const outerCanvasContainer = document.getElementById('fabric-canvas-wrapper');
//     const ratio          = canvas.getWidth() / canvas.getHeight();
//     const containerWidth = outerCanvasContainer.clientWidth;
//     const scale          = containerWidth / canvas.getWidth();
//     const zoom           = 1
//     canvas.setDimensions({width: containerWidth, height: containerWidth / ratio});
//     canvas.setViewportTransform([zoom, 0, 0, zoom, 0, 0]);
// }
// window.addEventListener('resize', resizeCanvas, true);

function move_all_to_back(group){
    var objs = group.getObjects();
    for (let j = 0, len =  group.size(); j < len; j++) {
            canvas.sendToBack(objs[j]);
    }
}

// document.getElementById('c').fabric = canvas;
fabric.Object.prototype.transparentCorners = false;
canvas.on('mouse:over', function(e) {
  if(e.target && !Frozen) {
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
      move_all_to_back(color_group)
      canvas.renderAll();
  }
});
canvas.on('mouse:out', function(e) {
  if(e.target && !Frozen){
    e.target.set('stroke', 'none');
    e.target.set('strokeWidth',0);
    e.target.set('left', e.target.get('left')+1.5);
    e.target.set('top', e.target.get('top')+1.5);
    canvas.renderAll();
  }
});


function loadImage() {
    // var canvas = document.getElementById("c").fabric;
    canvas.clear()
    image_group = new fabric.Group([]);
    color_group = new fabric.Group([]);
    img_array = [];
    color_array = [];
    img_opacity = 1;
    raw_width = -1;
    raw_height = -1;
    frozen = false;
    zoom = 1


    Frozen = true;
    myPromise = window.pywebview.api.loadImage()
    myPromise.then(
        function(value) {
            value = JSON.parse(value);
            var zoomb = document.getElementById('zoomBar');
            zoomb.value = 100;
            raw_width =  value[0].length *64;
            raw_height = value.length * 64
            canvas.setWidth(raw_width);
            canvas.setHeight(raw_height);
            canvas.calcOffset();
            image_group = new fabric.Group([]);
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
        Frozen = false;
        },
        function(error) { Frozen = false;}
        );

};

function evalImage(){
    img_opacity = .5
    Frozen = true;
    myPromise = window.pywebview.api.evalImage()
    myPromise.then(
        function(value) {
            color_array = []
            value = JSON.parse(value);
            var total = 0;
            var div = 0;
            for (let i = 0, len = value.length; i < len; i++) {
                color_array.push([])
                for (let j = 0, len = value[i].length; j < len; j++) {
                    total+=value[i][j];
                    div+=1;
                    img_array[i][j].score = value[i][j].toFixed(5)
                    img_array[i][j].set('opacity', img_opacity);
                    if(value[i][j] > .95){
                        var rect = new fabric.Rect({
                          left: j*64,
                          top: i*64,
                          fill: 'green',
                          width: 64,
                          height: 64
                        });
                    }else if (value[i][j] > 0.875) {
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
                    rect.selectable = false;
                    canvas.sendToBack(rect)
                    color_array[i].push(rect)
                    color_group.add(rect)
                }
            }
            avg_score = (total/div).toFixed(5)
            document.getElementById('total').innerHTML = avg_score;
            Frozen = false;
        },
        function(error) {Frozen = false;}
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
    canvas.renderAll();
}

function zoomBar(){
    var zoomb = document.getElementById('zoomBar')
    if(raw_width != -1 && raw_height != -1){
        zoom = zoomb.value/100.0
        canvas.setZoom(zoom);
        canvas.setWidth(raw_width * canvas.getZoom());
        canvas.setHeight(raw_height * canvas.getZoom());
    }
}

function changeTemplate() {
    Frozen = true;
    myPromise = window.pywebview.api.changeTemplate()
    myPromise.then(
        function(value) {
            document.getElementById('total').innerHTML = '';
            document.getElementById('template').innerHTML = value;
            document.getElementById('total').innerHTML = '';
            var objs = color_group.getObjects();
            for (let j = 0, len =  color_group.size(); j < len; j++) {
                canvas.remove(objs[j]);
            }
            var objs = color_group.getObjects();
            for (let j = 0, len =  image_group.size(); j < len; j++) {
                objs[j].score = 'n/a';
            }
            color_group = new fabric.Group([]);
            Frozen = false;
        },
        function(error) {Frozen = false;}
        );
}








