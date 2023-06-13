import os

import webview
from PIL import Image

from io import BytesIO
from base64 import b64encode
import json

import numpy as np

def get_tiles(iw):
    image = np.asarray(iw)
    length, width, c = image.shape
    tiles = []
    for x in range(0, length, 64):
        tiles.append([])
        for y in range(0, width, 64):
            if x + 64 < length and y + 64 < width:
                tiles[-1].append(np.copy(image[x:x + 64, y:y + 64, :]))
            elif x + 64 < length:
                temp = np.ones((64, 64, 3))*255
                tar = np.copy(image[x:x + 64, y:, :])
                temp[:tar.shape[0], :tar.shape[1], :] = tar
                tiles[-1].append(temp)
            elif y + 64 < width:
                temp = np.ones((64, 64, 3))*255
                tar = np.copy(image[x:, y:y + 64, :])
                temp[:tar.shape[0], :tar.shape[1], :] = tar
                tiles[-1].append(temp)
            elif x + 64 >= length and y + 64 >= width:
                temp = np.ones((64, 64, 3))*255
                tar = np.copy(image[x:, y:, :])
                temp[:tar.shape[0], :tar.shape[1], :] = tar
                tiles[-1].append(temp)
    return tiles

class Api:
    def addItem(self, title):
        print('Added item %s' % title)
    def loadImage(self):
        file_types = ('Image Files (*.bmp;*.jpg;*.gif)', 'All files (*.*)')
        result = webview.windows[0].create_file_dialog(webview.OPEN_DIALOG, allow_multiple=True, file_types=file_types)
        img = Image.open(result[0]).convert('RGB')
        tiles = get_tiles(img)
        new_tiles = []
        for x in tiles:
            tile_row = []
            for y in x:

                image = Image.fromarray(y.astype(np.uint8))
                image_io = BytesIO()
                image.save(image_io, 'PNG')
                dataurl = 'data:image/png;base64,' + b64encode(image_io.getvalue()).decode('ascii')
                tile_row.append(dataurl)
            new_tiles.append(tile_row)
        # print(str(new_tiles[0][0]))
        # return str(new_tiles[0][0])
        return json.dumps(new_tiles)
    def removeItem(self, item):
        print('Removed item %s' % item)

    def editItem(self, item):
        print('Edited item %s' % item)

    def toggleItem(self, item):
        print('Toggled item %s' % item)

    def toggleFullscreen(self):
        webview.windows[0].toggle_fullscreen()

    def test(self):
        print("test")


if __name__ == '__main__':
    api = Api()
    webview.create_window('The Paleographer\'s Eye From the Machine', 'assets/new_index.html', js_api=api, min_size=(600, 500))
    webview.start(gui='qt')
