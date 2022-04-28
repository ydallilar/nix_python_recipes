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
    paramsd = {}
    for i in range(len(input_params)):
        paramsd[input_params[i].name] = input_params[i].value
    return paramsd

SETUPS = {"SPIFFIER" : {
            "250mas" : {"crpix1" : 32, "crpix2" : 32, "sign" : -1, "pxscl" : 125},
            "100mas" : {"crpix1" : 32, "crpix2" : 32, "sign" : -1, "pxscl" : 50},
            "25mas" : {"crpix1" : 32, "crpix2" : 32, "sign" : -1, "pxscl" : 25},
            },
         "NIX" : {
            "13mas-JHK" : {"crpix1" : 1064.9, "crpix2" : 1099.1, "sign" : 1, "pxscl" : 13},
            "27mas-JHK" : {"crpix1" : 958.0, "crpix2": 1065.9, "sign" : -1, "pxscl" : 27},
            "13mas-LM" : {"crpix1" : 1064.9, "crpix2" : 1099.1, "sign" : 1, "pxscl" : 13},
            }
         }

def get_config(hdul):

    INS = hdul[0].header["ESO INS1 SCSM NAME"]
    if INS == "NIX":
        config = SETUPS[INS][hdul[0].header["ESO INS2 NXCW NAME"]]
    elif INS == "SPIFFIER":
        config = SETUPS[INS][hdul[0].header["ESO INS3 SPXW NAME"]]
    else:
        raise ValueError("%s not a valid instrument" % INS)

    return config

class TestRecipe(esorexplugin.RecipePlugin):

    name = "eris_nix_img_wcs_user"

    version = VersionNumber(0, 1, 0)

    synopsis = "Tunable WCS transformation of NIX, can also apply user defined offsets if given."

    description = "..."

    author = "Yigit Dallilar"
    email = "ydallilar@mpe.mpg.de"

    #copyright = EsoCopyright("TEST_INSTRUMENT", [2010, 2013, 2014, 2015, 2017])

    ENV_ENABLED=False

    parameters = [
            ValueParameter("nix.pxscl", 13.085, description="Update NIX pixel scale [mas/pix]", env_enabled=ENV_ENABLED),
            ValueParameter("nix.padelta", 0.313, description="Delta angle offset [deg]", env_enabled=ENV_ENABLED),
            ValueParameter("nix.offset_angle_from_north", -2.327, description="Relative AO offset angle from north [deg]", env_enabled=ENV_ENABLED),
            ValueParameter("nix.offset_scale", 0.998, description="Scaling of the offsets, ie. 1. for perfect scaling", env_enabled=ENV_ENABLED),
            ValueParameter("nix.ndx_origin", 13, description="Index of the origin frame", env_enabled=ENV_ENABLED),
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

        global params
        params = dict_params(self.input_parameters)

        pxscl = params["nix.pxscl"]
        padelta = params["nix.padelta"]
        offN = params["nix.offset_angle_from_north"]
        offscl = params["nix.offset_scale"]
        orig = params["nix.ndx_origin"]

        USER_OFF_ENABLED = False

        if orig > len(frames) - 1:
            raise ValueError("nix.ndx_origin{=%d} can't be larger than number of frames {=%d}." % (orig, len(frames)))

        # See if user offset given
        match = [i for i, frame in enumerate(frames) if frame.tag == "USER_OFFSET"]
        if  len(match) == 1:
            userf = frames.pop(match[0])
            user_off = np.loadtxt(userf.filename)
            user_off = np.cumsum(user_off, axis=0)
            USER_OFF_ENABLED = True
            logger.info("Using user offsets from: %s" % userf.filename)
        else:
            logger.info("No user offsets detected using defaults.")

        hdul_l = [frame.open() for frame in frames]

        ra_o = hdul_l[orig][0].header["RA"]
        dec_o = hdul_l[orig][0].header["DEC"]
        
        proj_f = np.cos(np.deg2rad(dec_o))
        PA = hdul_l[0][0].header["ESO ADA POSANG"] + padelta
        conf = get_config(hdul_l[0])
        cdu_d = np.array([[conf["sign"], 0], [0, 1]])
        cdu_s = np.array([[-1, 0], [0, 1]])
        cd = np.matmul(rotM(PA), cdu_d)*pxscl*1e-3/60/60.

        output_frames = []

        for i in range(len(hdul_l)):
            
            hdul_l[i][0].header["CUNIT1"] = "deg"
            hdul_l[i][0].header["CUNIT2"] = "deg"
            hdul_l[i][0].header["CTYPE1"] = "RA---TAN"
            hdul_l[i][0].header["CTYPE2"] = "DEC--TAN"
            hdul_l[i][0].header["CRPIX1"] = conf["crpix1"]
            hdul_l[i][0].header["CRPIX2"] = conf["crpix2"]
            hdul_l[i][0].header["CD1_1"] = cd[0][0]
            hdul_l[i][0].header["CD1_2"] = cd[0][1]
            hdul_l[i][0].header["CD2_1"] = cd[1][0]
            hdul_l[i][0].header["CD2_2"] = cd[1][1]

            if USER_OFF_ENABLED:
                user_off_t = np.matmul(rotM(offN),user_off[i,:])/60./60.*offscl
                hdul_l[i][0].header["CRVAL1"] = ra_o + user_off_t[0]/proj_f
                hdul_l[i][0].header["CRVAL2"] = dec_o + user_off_t[1]
            else:
                hdul_l[i][0].header["CRVAL1"] = hdul_l[i][0].header["RA"]
                hdul_l[i][0].header["CRVAL2"] = hdul_l[i][0].header["DEC"]
            
            new_frame = Frame("%s" % frames[i].filename.split("/")[-1], "XYZ", type = Frame.TYPE_IMAGE)
            output_frames.append(new_frame)
            new_frame.write(hdul_l[i], overwrite=True)
            
        return output_frames
