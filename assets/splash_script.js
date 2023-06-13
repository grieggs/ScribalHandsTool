document.addEventListener('DOMContentLoaded', function() {
    setTimeout(function(){
           myPromise = window.pywebview.api.getTemplates();
           myPromise.then(
                function(value) {
                    var value = JSON.parse(value);
                    var select = document.getElementById('template');
                    for (let i = 0, len = value.length; i < len; i++) {
                        var opt = document.createElement('option');
                        opt.value = value;
                        opt.innerHTML = value;
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
};