var pyastrovis = require('./index');
var base = require('@jupyter-widgets/base');

module.exports = {
  id: 'pyastrovis',
  requires: [base.IJupyterWidgetRegistry],
  activate: function(app, widgets) {
      widgets.registerWidget({
          name: 'pyastrovis',
          version: pyastrovis.version,
          exports: pyastrovis
      });
  },
  autoStart: true
};

