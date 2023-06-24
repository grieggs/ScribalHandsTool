document.addEventListener('DOMContentLoaded', function() {
    setTimeout(function(){
           myPromise = window.pywebview.api.getTemplates();
           myPromise.then(
                function(value) {
                    var value = JSON.parse(value);
                    var select = document.getElementById('template');
                    for (let i = 0, len = value.length; i < len; i++) {
                        var opt = document.createElement('option');
                        opt.value = value[i];
                        opt.innerHTML = value[i];
                        select.appendChild(opt);
                    }
                    select.disabled = false;
                },
               function(error) {});
            }, 100);

}, false);

function openImageViewer(){
    var select = document.getElementById('template');
    myPromise = window.pywebview.api.openImageViewer(select.value);
}

function loadingScreenOn(){
    var panel = document.getElementById('panel');
    var loadingpanel = document.getElementById('loading-pane');
    panel.style.visibility = "hidden";
    loadingpanel.style.visibility = "visible";
}

function loadingScreenOff(){
    var panel = document.getElementById('panel');
    var loadingpanel = document.getElementById('loading-pane');
    panel.style.visibility = "visible";
    loadingpanel.style.visibility = "hidden";
}


function buildTemplate(){
    myPromise = window.pywebview.api.buildTemplate();
    loadingScreenOn()
    myPromise.then(
        function(value) {
            loadingScreenOff();

        },
        function(error) {loadingScreenOff();}
    );

    var select = document.getElementById('template');
    var i, L = select.options.length - 1;
    for(i = L; i >= 0; i--) {
        if(select[i].disabled == false){
            select.remove(i);
        }
    }
    myPromise = window.pywebview.api.getTemplates();
    myPromise.then(
        function(value) {
            var value = JSON.parse(value);
            for (let i = 0, len = value.length; i < len; i++) {
                var opt = document.createElement('option');
                opt.value = value[i];
                opt.innerHTML = value[i];
                select.appendChild(opt);
            }
            select.disabled = false;
        },
       function(error) {});
}