import av
import asyncio
import json
import aiohttp_cors
import numpy as np

from aiohttp import web
from asyncio import Queue
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack


class WebRTCStream(web.Application):

    class VideoStream(VideoStreamTrack):
        def __init__(self, buffer_queue_size):
            super(VideoStreamTrack, self).__init__()
            self.q = Queue(buffer_queue_size)
            self.loop = asyncio.get_running_loop()

        async def add_data(self, data, format):
            copy = await self.loop.run_in_executor(None, np.copy, data)
            await self.q.put((copy, format))

        async def recv(self):
            data, format = await self.q.get()
            frame = await self.loop.run_in_executor(None, av.VideoFrame.from_ndarray, data, format)
            pts, time_base = await self.next_timestamp()
            frame.pts = pts
            frame.time_base = time_base
            return frame

    def __init__(self, buffer_queue_size, *args, **kwargs):
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
        self.buffer_queue_size = buffer_queue_size

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

        stream = WebRTCStream.VideoStream(self.buffer_queue_size)
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
