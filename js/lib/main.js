var widgets = require('@jupyter-widgets/base');
var rtc = require('./webrtc.js');
var _ = require('lodash');


function getElementCSSSize(el) {
    var cs = getComputedStyle(el);
    var w = parseInt(cs.getPropertyValue("width"), 10);
    var h = parseInt(cs.getPropertyValue("height"), 10);
    return {width: w, height: h}
}

// Custom Model. Custom widgets models must at least provide default values
// for model attributes, including
//
//  - `_view_name`
//  - `_view_module`
//  - `_view_module_version`
//
//  - `_model_name`
//  - `_model_module`
//  - `_model_module_version`
//
//  when different from the base class.

// When serialiazing the entire widget state for embedding, only values that
// differ from the defaults will be specified.
var WebRTCModel = widgets.DOMWidgetModel.extend({
    defaults: _.extend(widgets.DOMWidgetModel.prototype.defaults(), {
        _model_name : 'WebRTCModel',
        _view_name : 'WebRTCView',
        _model_module : 'pyastrovis',
        _view_module : 'pyastrovis',
        _model_module_version : '0.1.0',
        _view_module_version : '0.1.0',
        url : '',
        id : '',
        width : 0,
        height : 0,
        position: [0,0]
    })
});


// Custom View. Renders the widget model.
var WebRTCView = widgets.DOMWidgetView.extend({

    render: function() {
        let url = this.model.get('url');
        let id = this.model.get('id');
        let height = this.model.get('height');
        let width = this.model.get('width');
        let video = document.createElement('video');
        let that = this;
        video.id=id;
        video.height=height;
        video.width=width;
        video.autoplay=true;
        let rc = rtc.create_rtc(video);
        rtc.negotiate_rtc(rc, id, url);
        this.el.appendChild(video);

        video.addEventListener("click",
            function mouse_handler(event) {
                var size = getElementCSSSize(this);
                var scaleX = this.videoWidth / size.width;
                var scaleY = this.videoHeight / size.height;

                var rect = this.getBoundingClientRect();  // absolute position of element
                var x = ((event.clientX - rect.left) * scaleX + 0.5)|0;
                var y = ((event.clientY - rect.top ) * scaleY + 0.5)|0;

                that.model.set('position', [x,y]);
                that.touch();
            }, false);

        //this.value_changed();
        //this.model.on('change:value', this.value_changed, this);
    },

    value_changed: function() {
        //this.el.textContent = this.model.get('value');
    }
});


module.exports = {
    WebRTCModel : WebRTCModel,
    WebRTCView : WebRTCView
};
