import esorexplugin
from esorexplugin import *

import numpy as np

from astropy.io import fits
import logging, sys, warnings

# Utils
logging.basicConfig(format='[%(levelname)s] %(message)s', level=logging.INFO,
                     stream=sys.stdout)
logger = logging.getLogger(__name__)

rotM = lambda ang : np.array([[np.cos(np.deg2rad(ang)), -np.sin(np.deg2rad(ang))],
                                    [np.sin(np.deg2rad(ang)), np.cos(np.deg2rad(ang))]])

params = {}

def dict_params(input_params):
    global params
    for i in range(len(input_params)):
        params[input_params[i].name] = input_params[i].value
    return params

def combine_frames(frames, st_ndx, end_ndx):

    hduls = [frame.open() for frame in frames[st_ndx:end_ndx]]

    hdul = hduls[0].copy()

    hdul[1].data = np.sum(np.stack([hdul[1].data/hdul[2].data**2 for hdul in hduls], axis=0), axis=0)/\
                        np.sum(np.stack([1./hdul[2].data**2 for hdul in hduls], axis=0), axis=0)  
    
    hdul[1].data[np.where(np.isnan(hdul[1].data))] = 0
    hdul[2].data = np.sqrt(np.mean(np.stack([hdul[2].data**2 for hdul in hduls], axis=0), axis=0))

    return hdul


class TestRecipe(esorexplugin.RecipePlugin):

    name = "eris_nix_img_nexpo_merge"

    version = VersionNumber(0, 1, 0)

    synopsis = "Merge NEXPO frames"

    description = "..."

    author = "Yigit Dallilar"
    email = "ydallilar@mpe.mpg.de"

    #copyright = EsoCopyright("TEST_INSTRUMENT", [2010, 2013, 2014, 2015, 2017])

    ENV_ENABLED=False

    parameters = [
            ValueParameter("nix.nexpo", 1., description="NEXPO", env_enabled=ENV_ENABLED),
        ]
    
    # No clue what this is
    recipeconfig = [
            FrameData('RAW', min = 2, max = 5,
                      inputs = [
                            FrameData('MASTER'),
                            FrameData('CALIB', min = 0, max = 2)
                        ],
                      outputs = ['PROD']),
            FrameData('MASTER', inputs = [FrameData('RAW')]),
            FrameData('CALIB')
        ]

    # again no clue
    def set_frame_group(self, frame):
        if frame.tag == 'RAW':
            frame.group = Frame.GROUP_RAW
        else:
            frame.group = Frame.GROUP_CALIB
        frame.level = Frame.LEVEL_FINAL

    def process(self, frames, *args):

        dict_params(self.input_parameters)

        sz = len(frames)
        nexpo = int(params["nix.nexpo"])

        output_frames = []

        for i in range(sz//nexpo):
            
            hdul = combine_frames(frames, i*nexpo, (i+1)*nexpo)
            new_frame = Frame("merge.%s" % frames[i*nexpo].filename.split("/")[-1], "XYZ", type = Frame.TYPE_IMAGE)
            output_frames.append(new_frame)
            new_frame.write(hdul, overwrite=True)

        return output_frames
