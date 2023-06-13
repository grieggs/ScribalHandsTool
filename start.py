import os

import webview
from PIL import Image
from models.models import ArcResnet18 as ScribalHandClassifier
from io import BytesIO
from base64 import b64encode
import json
import torch
import numpy as np
import glob


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


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

def eval_image(tiles, template, model):
    preds = np.zeros((len(tiles), len(tiles[0])))
    for i, x in enumerate(tiles):
        for j, y in enumerate(x):
            input = torch.tensor(y/255)
            input = input.permute(2, 0, 1).float()
            input = input.unsqueeze(0)
            pred, junk = model(input)
            preds[i, j] = torch.nn.functional.cosine_similarity(pred, template, dim=-1)
    return preds



class SplashApi:
    def __init__(self):
        self.templates = {}
    def getTemplates(self):
        templates = glob.glob('templates/*.npy')
        for x in templates:
            self.templates[os.path.basename(x)] = x
        return json.dumps(list(self.templates.keys()))

    def openImageViewer(self, template):
        api = Api("models/shm.ckpt", self.templates[template])
        webview.create_window('The Paleographer\'s Eye From the Machine', 'assets/docpage.html', js_api=api,
                              min_size=(600, 500))


class Api:
    def __init__(self, chk_path, template_path):
        self.done = False
        self.chk_path = chk_path
        self.template_path = template_path
        checkpoint = torch.load(chk_path, map_location=torch.device('cpu'))
        training_outs = checkpoint['state_dict']['metric_fc.weight'].shape[0]
        torch.set_grad_enabled(False)
        self.model = ScribalHandClassifier(num_classes=training_outs).cpu()
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()
        self.model = self.model.cpu()
        self.template = np.load(template_path,allow_pickle=True).item()
        self.tiles = None
        self.loaded_image = None

    def addItem(self, title):
        print('Added item %s' % title)
    def loadImage(self):
        file_types = ('Image Files (*.bmp;*.jpg;*.gif)', 'All files (*.*)')
        result = webview.windows[0].create_file_dialog(webview.OPEN_DIALOG, allow_multiple=True, file_types=file_types)
        img = Image.open(result[0]).convert('RGB')
        self.loaded_image = result[0]
        self.tiles = get_tiles(img)
        new_tiles = []
        for x in self.tiles:
            tile_row = []
            for y in x:
                image = Image.fromarray(y.astype(np.uint8))
                image_io = BytesIO()
                image.save(image_io, 'PNG')
                dataurl = 'data:image/png;base64,' + b64encode(image_io.getvalue()).decode('ascii')
                tile_row.append(dataurl)
            new_tiles.append(tile_row)
        return json.dumps(new_tiles)

    def evalImage(self):
        total = 0
        template = np.zeros(512)
        for x in self.template:
            if x != os.path.basename(self.loaded_image):
                ref = self.template[x]
                samples = ref.shape[0] * ref.shape[1]
                total += samples
                template += ref.sum(axis=0).sum(axis=0)
        template = template / total
        template = torch.tensor(template)
        preds = eval_image(self.tiles, template, self.model)
        return json.dumps(preds, cls=NumpyEncoder)

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

class Manager:
    def __init__(self):
        self.done = False


if __name__ == '__main__':

    # for x in tqdm_gui(range(1,1000000)):
    #     print('hello world')
    # manager = Manager()
    # # while not manager.done:
    # one = range(0,10)
    # two = range(0,10)
    # pbar = tqdm_gui(total=100)
    # for i in one:
    #     for j in two:
    #         time.sleep(0.05)
    #


    # splash

    # for outer in tqdm.tqdm([10, 20, 30, 40, 50], desc=" outer", position=0):
    #     for inner in tqdm.tqdm(range(outer), desc=" inner loop", position=1, leave=False):
    #         time.sleep(0.05)
    # print("done!")

    # api = Api("models/shm.ckpt", '/home/sgrieggs/PycharmProjects/ScribalHandsTool/templates/Hoccleve.npy')
    # webview.create_window('The Paleographer\'s Eye From the Machine', 'assets/docpage.html', js_api=api,
    #                       min_size=(600, 500))
    api = SplashApi()
    webview.create_window('test', 'assets/index.html', js_api=api, min_size=(600, 500))
    webview.start(gui='qt')
        #
        # print("test")


    # webview.create_window('The Paleographer\'s Eye From the Machine', 'assets/new_index.html', js_api=api, min_size=(600, 500))
    #
    # webview.start(gui='qt')
