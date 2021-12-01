import os
import math
import struct
import aiofiles
import asyncio
import itertools
import concurrent.futures

import numpy as np

from astropy.io import fits
from skimage.util import crop


def split_list(alist, wanted_parts=1):
    length = len(alist)
    return [alist[i*length // wanted_parts: (i+1)*length // wanted_parts]
            for i in range(wanted_parts)]


async def _header_size_obj(file_obj):
    num_bytes = 0
    row_size = 80

    while True:
        row = await file_obj.read(row_size)
        if not row:
            raise Exception()
        row = row.decode('ascii')
        num_bytes += row_size
        if row.replace(' ', '') == 'END':
            break

    while await file_obj.tell() % 2880 != 0:
        await file_obj.read(row_size)
        num_bytes += row_size

    return num_bytes


def _get_spectral_line(filename, pixel_x, pixel_y, width, byte_depth,
                       header_size_bytes, image_size_bytes, channels,
                       bscale, bzero, format):

    file_no = None
    try:
        file_no = os.open(filename, os.O_RDONLY)
        result = [0]*len(channels)
        for index, channel in enumerate(channels):
            offset = header_size_bytes + (image_size_bytes * channel) + (byte_depth * ((pixel_x * width) + pixel_y))
            os.lseek(file_no, offset, 0)
            val = struct.unpack(format, os.read(file_no, byte_depth))[0]
            if not (bscale == 1.0 and bzero == 0.0):
                val = (val * bscale) + bzero
            result[index] = val
        return result
    finally:
        if file_no:
            os.close(file_no)


class FITSImageCubeStream(object):

    def __init__(self, filename, file_obj, header, header_size_bytes, num_processes, loop, fits_special_dim='NAXIS4'):
        self.loop = loop
        self.file_obj = file_obj
        #self.file_list = file_list
        self.filename = filename
        self.header = header
        self.header_size_bytes = header_size_bytes
        self.width = int(header['NAXIS1'])
        self.height = int(header['NAXIS2'])

        self.num_channels = int(header[fits_special_dim])
        
        self.bscale = self.header.get('BSCALE', 1)
        self.bzero = self.header.get('BZERO', 0)

        channels = [i for i in range(0, self.num_channels-1)]
        self.channel_split = split_list(channels, num_processes)
        self.pool = concurrent.futures.ProcessPoolExecutor(num_processes)

        self.bitpix = int(header['BITPIX'])
        if self.bitpix == 8:
            self.dtype = 'B'
            self.format = 'B'
            self.byte_depth = 1
        elif self.bitpix == 16:
            self.dtype = '>i2'
            self.format = '>h'
            self.byte_depth = 2
        elif self.bitpix == 32:
            self.dtype = '>i4'
            self.format = '>i'
            self.byte_depth = 4
        elif self.bitpix == -32:
            self.dtype = '>f4'
            self.format = '>f'
            self.byte_depth = 4
        elif self.bitpix == -64:
            self.dtype = '>f8'
            self.format = '>d'
            self.byte_depth = 8
        else:
            raise Exception('unknown bitpix type')

        self.image_size_bytes = int(self.width * self.height * self.byte_depth)
        print(self.width, self.height, self.image_size_bytes, self.byte_depth)
        self.x = self.width
        self.y = self.height

    async def close(self):
        await self.file_obj.close()
        #for i in self.file_list:
        #    await self.loop.run_in_executor(None, os.close, i)
        await self.loop.run_in_executor(None, self.pool.shutdown, True)

    def get_dtype(self):
        return self.dtype

    def get_x(self):
        return self.x

    def get_y(self):
        return self.y

    def get_header(self):
        return self.header

    def get_num_channels(self):
        return self.num_channels

    async def get_spectral_line(self, pixel_x, pixel_y):
        if 0 < pixel_x > self.width:
            raise Exception("invalid x pixel index")

        if 0 < pixel_y > self.height:
            raise Exception("invalid y pixel index")

        tasks = []
        for i, j in enumerate(self.channel_split):
            task = self.loop.run_in_executor(self.pool, _get_spectral_line, self.filename,
                                             pixel_x, pixel_y, self.width, self.byte_depth,
                                             self.header_size_bytes, self.image_size_bytes, j,
                                             self.bscale, self.bzero, self.format)
            tasks.append(task)

        result = await asyncio.gather(*tasks)
        return list(itertools.chain(*result))

    def _convert_data(self, data, x1, x2, y1, y2):
        _x1 = x1
        _y1 = y1
        _x2 = x2
        _y2 = y2

        arr = np.frombuffer(data, dtype=self.dtype)
        image_buff = np.lib.stride_tricks.as_strided(arr, (self.y, self.x))
        image_buff = crop(image_buff, ((_x1, _x2), (_y1, _y2)))

        if not (self.bscale == 1.0 and self.bzero == 0.0):
            image_buff = (image_buff * self.bscale) + self.bzero
        return image_buff


    async def get_channel_data(self, channel, x1=0, x2=0, y1=0, y2=0):
        if not isinstance(channel, int):
            raise Exception('channel not an int')

        if 0 > channel:
            raise Exception('invalid channel')

        if channel >= self.num_channels:
            raise Exception('invalid channel')

        #await self.file_obj.seek(0)
        offset = self.header_size_bytes+(self.image_size_bytes*channel)
        await self.file_obj.seek(offset)
        data = await self.file_obj.read(self.image_size_bytes)
        image_buff = await self.loop.run_in_executor(None, self._convert_data, data, x1, x2, y1, y2)
        #print(channel, image_buff)
        return image_buff

    @staticmethod
    async def open(filename, buffering=1, num_processes=2):
        loop = asyncio.get_running_loop()

        with concurrent.futures.ThreadPoolExecutor() as pool:
            header = await loop.run_in_executor(pool, fits.getheader, filename, 0)

        #file_list = [await loop.run_in_executor(None, os.open, filename, os.O_RDONLY) for _ in range(num_processes)]
        #file_obj = await aiofiles.open(filename, mode='rb', buffering=buffering)
        file_obj = await aiofiles.open(filename, mode='rb', buffering=buffering)
        header_size_bytes = await _header_size_obj(file_obj)
        print(header_size_bytes)

        fits_stream = FITSImageCubeStream(filename, file_obj, header, header_size_bytes, num_processes, loop)
        return fits_stream
