import uuid
import asyncio
import ipywidgets as widgets

from traitlets import Unicode, Int, Tuple
from aiohttp import web

from .webrtc import WebRTCStream


@widgets.register
class WebRTCClientWidget(widgets.DOMWidget):

    _view_name = Unicode('WebRTCView').tag(sync=True)
    _model_name = Unicode('WebRTCModel').tag(sync=True)
    _view_module = Unicode('pyastrovis').tag(sync=True)
    _model_module = Unicode('pyastrovis').tag(sync=True)
    _view_module_version = Unicode('^0.1.0').tag(sync=True)
    _model_module_version = Unicode('^0.1.0').tag(sync=True)
    url = Unicode('').tag(sync=True)
    id = Unicode('').tag(sync=True)
    width = Int(512).tag(sync=True)
    height = Int(512).tag(sync=True)
    position = Tuple(default_value=(0, 0)).tag(sync=True)

    def __init__(self, widget, url, width, height):
        super().__init__()
        self.url = url
        self.id = str(uuid.uuid4())
        self.widget = widget
        self.width = width
        self.height = height
        self.position = (0,0)

    def get_position(self):
        return self.position

    async def add_data(self, data, format='rgb24'):
        await self.widget.app.add_data(self.id, data, format)

    async def close(self):
        await self.widget.app.close_endpoint(self.id)


class WebRTCWidget(object):
    def __init__(self, app, runner, url):
        self.app = app
        self.runner = runner
        self.url = url

    def create_panel(self, width=512, height=512):
        return WebRTCClientWidget(self, self.url, width, height)

    async def close(self):
        await self.app.close()
        await self.runner.cleanup()

    @classmethod
    async def create_server(cls, host='localhost', port=8080,
                            client_url='http://localhost:8080',
                            buffer_queue_size=1):
        app = WebRTCStream(buffer_queue_size)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, host, port)
        await site.start()
        return WebRTCWidget(app, runner, client_url)

    @staticmethod
    def wait_for_change(widget, value):
        future = asyncio.Future()
        def getvalue(change):
            future.set_result(change.new)
            widget.unobserve(getvalue, value)
        widget.observe(getvalue, value)
        return future

