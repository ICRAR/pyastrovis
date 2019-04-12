import asyncio
import json
import av
import aiohttp_cors

from concurrent.futures import CancelledError
from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from aiortc.mediastreams import MediaStreamError, VIDEO_TIME_BASE, VIDEO_PTIME, VIDEO_CLOCK_RATE


class WebRTCStream(web.Application):

    class VideoStream(VideoStreamTrack):
        def __init__(self):
            super(VideoStreamTrack, self).__init__()
            self.loop = asyncio.get_running_loop()
            self.lock = asyncio.Lock()
            self.data = None
            self.format = None
            self.frame = av.VideoFrame(width=640, height=480)
            for p in self.frame.planes:
                p.update(bytes(p.buffer_size))

        async def add_data(self, data, format):
            async with self.lock:
                self.data = data
                self.format = format

        async def recv(self):
            try:
                pts, time_base = await self.next_timestamp()
                async with self.lock:
                    if self.data is None:
                        self.frame.pts = pts
                        self.frame.time_base = time_base
                        return self.frame
                    self.frame = av.VideoFrame.from_ndarray(array=self.data, format=self.format)
                    self.data = None
                    self.frame.pts = pts
                    self.frame.time_base = time_base
                    return self.frame

            except CancelledError:
                raise MediaStreamError
            except Exception:
                pts, time_base = await self.next_timestamp()
                frame = av.VideoFrame(width=640, height=480)
                for p in frame.planes:
                    p.update(bytes(p.buffer_size))
                frame.pts = pts
                frame.time_base = time_base
                return frame

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        cors = aiohttp_cors.setup(self, defaults={
            "*": aiohttp_cors.ResourceOptions(
                allow_credentials=True,
                expose_headers="*",
                allow_headers="*",
            )})

        self.router.add_post('/offer', self.offer)
        for route in list(self.router.routes()):
            cors.add(route)

        self.pcs = []
        self.stream_tracks = {}

    async def add_data(self, id, data, format):
        track = self.stream_tracks.get(id, None)
        if track:
            await track[0].add_data(data, format)

    async def close_endpoint(self, id):
        track = self.stream_tracks.get(id, None)
        if track:
            await track[2].close()

    async def offer(self, request):
        params = await request.json()
        conn_id = request.headers['ID']

        offer = RTCSessionDescription(
            sdp=params['sdp'],
            type=params['type'])

        pc = RTCPeerConnection()
        self.pcs.append(pc)

        @pc.on('iceconnectionstatechange')
        async def on_iceconnectionstatechange():
            if pc.iceConnectionState == 'failed':
                await pc.close()
                self.pcs.remove(pc)

        stream = WebRTCStream.VideoStream()
        sender = pc.addTrack(stream)
        self.stream_tracks[conn_id] = (stream, sender, pc)

        @sender.transport.on('statechange')
        async def change_state():
            if sender.transport.state == 'closed':
                await pc.close()
                self.pcs.remove(pc)

                for key, value in dict(self.stream_tracks).items():
                    if value[1] == sender:
                        del self.stream_tracks[key]
                        break

        await pc.setRemoteDescription(offer)
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        return web.Response(
            content_type='application/json',
            text=json.dumps({
                'sdp': pc.localDescription.sdp,
                'type': pc.localDescription.type
            }))

    async def close(self):
        await self.shutdown()
        await self.cleanup()

        for pc in list(self.pcs):
            await pc.close()

        self.stream_tracks.clear()
        self.pcs.clear()
