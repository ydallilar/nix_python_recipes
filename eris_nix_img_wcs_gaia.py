import esorexplugin
from esorexplugin import *

import numpy as np

# One will typically also want to import the astropy FITS I/O module to be able
# to load and save the recipe input frames.
from scipy.signal import fftconvolve
from scipy.optimize import leastsq
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astroquery.vizier import Vizier
from astropy.wcs import WCS
from astropy.utils.exceptions import AstropyWarning
from sep import extract
import astropy.units as u
import logging, sys, warnings
import numpy.lib.recfunctions as rfn

logging.basicConfig(format='[%(levelname)s] %(message)s', level=logging.INFO,
                     stream=sys.stdout)
logger = logging.getLogger(__name__)

Vizier.ROW_LIMIT = -1
#Vizier.SERVER = "http://vizier.cfa.harvard.edu/"

# helper functions
def gaussian(p, x, y):
    return p[0]*np.exp(-((x-p[1])**2+(y-p[2])**2)*0.5/p[3]**2)/(2*np.pi)

rotM = lambda ang : np.array([[np.cos(np.deg2rad(ang)), -np.sin(np.deg2rad(ang))],
                                    [np.sin(np.deg2rad(ang)), np.cos(np.deg2rad(ang))]])

def dict_params(input_params):
    paramsd = {}
    for i in range(len(input_params)):
        paramsd[input_params[i].name] = input_params[i].value
    return paramsd

def wcs_sf(sources, ra_f, dec_f, ang, pxscl, sign):

    crpix = np.array([1024., 1024.])
    crval = np.array([ra_f, dec_f])
    xy = np.array(list(zip(sources["x"], sources["y"])))

    cdu = np.array([[sign, 0], [0, 1]])
    cd = np.matmul(rotM(ang), cdu)*np.abs(pxscl)*1e-3/60/60.

    res = np.zeros((len(sources), 2))
    
    pix_off = xy - crpix
    #pix_off[:,0] /= (np.cos(np.deg2rad(sources["DEC"])))
    
    for i in range(len(sources)):
        res[i,:] = np.matmul(cd, pix_off[i,:])
        res[i,0] /= (np.cos(np.deg2rad(sources["DEC"][i]))) 
        res[i,:] += crval

    return res

def wcserr_sf(p, sources, sign):

    g_coo = np.array(list(zip(sources["GRA"], sources["GDEC"])))
    n_coo = wcs_sf(sources, p[0], p[1], p[2], p[3], sign)
    return np.sqrt(((g_coo[:,0]-n_coo[:,0])*(np.cos(np.deg2rad(g_coo[:,1]))))**2+(g_coo[:,1]-n_coo[:,1])**2)

class GAIAProd:

    gaia_cols = ["Source", "RA_ICRS", "e_RA_ICRS", "DE_ICRS", "e_DE_ICRS", "Gmag", "pmRA", "pmDE"]
    gaia_fmt = ["A30", "D", "D", "D", "D", "D", "D", "D"]
    
    def __init__(self, coo, date, params):
        self.coo = coo
        self.date = date
        self.params = params

    def fetch_catalog(self, box_width, maglim):

        maglim = self.params["gaia.maglim"]
        prop_motion = self.params["gaia.prop_motion"]

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', AstropyWarning)
            result = Vizier.query_region(self.coo, width=box_width, catalog="I/350/gaiaedr3", column_filters={'Gmag': '<%.f' % maglim})
        gaia_table = result[0][tuple(self.gaia_cols)]
        if prop_motion:
            gaia_table["RA_ICRS"] += (gaia_table["pmRA"]*(self.date-57388.)/np.cos(np.deg2rad(gaia_table["DE_ICRS"]))*1e-3/60./60./365.25)
            gaia_table["DE_ICRS"] += (gaia_table["pmDE"]*(self.date-57388.)*1e-3/60./60./365.25)
        cols = [fits.Column(name=name, format=fmt, array=gaia_table[name]) for name, fmt in zip(self.gaia_cols, self.gaia_fmt)]
        self.catalog = fits.BinTableHDU.from_columns(cols)

    def mock_image(self, pa, pxscl, fov, sign):

        zeromag = self.params["gaia.zeromag"]
        fwhm = self.params["gaia.fwhm"]*1e3

        sz = np.int64(fov/pxscl)
        extnd = int(fwhm/pxscl*10)
        im = np.zeros([sz+2*extnd, sz+2*extnd])

        X = np.arange(im.shape[0])
        XX, YY = np.meshgrid(X, X)
        
        cdu = np.array([[sign, 0], [0, 1]])*pxscl*1e-3/60./60
        cd = np.matmul(rotM(pa), cdu)
        wcs = WCS({"CTYPE1" : "RA---TAN", "CTYPE2" : "DEC--TAN", 
            "CRVAL1" : self.coo.ra.value, "CRVAL2" : self.coo.dec.value,
            "CRPIX1" : im.shape[0]//2, "CRPIX2" : im.shape[0]//2,
            "CD1_1" : cd[0][0], "CD1_2" : cd[0][1], "CD2_1" : cd[1][0], "CD2_2" : cd[1][1]})

        for target in self.catalog.data:
        
            x, y = wcs.world_to_pixel(SkyCoord(target["RA_ICRS"]*u.deg, target["DE_ICRS"]*u.deg))
            cnts = 10**((zeromag-target["Gmag"])/2.5)

            x_im, y_im = int(x), int(y)

            im[y_im-extnd:y_im+extnd,
                x_im-extnd:x_im+extnd] += gaussian([cnts, x, y, fwhm/pxscl/2.35], XX[y_im-extnd:y_im+extnd,x_im-extnd:x_im+extnd], 
                                                                            YY[y_im-extnd:y_im+extnd,x_im-extnd:x_im+extnd])


        self.hdu = fits.ImageHDU(im, header=wcs.to_header())
        self.wcs = wcs

    def relative_offset(self, image):

        search_box = 400
        conv = fftconvolve(self.hdu.data[::-1,::-1], image, mode="same").T
        fits.PrimaryHDU(self.hdu.data).writeto("test.fits", overwrite=True)

        c_x = self.hdu.data.shape[0]//2
        shft = np.unravel_index(np.argmax(conv[c_x-search_box:c_x+search_box,
                                            c_x-search_box:c_x+search_box]), (2*search_box, 2*search_box)) - np.array(*[search_box])
        pos = -shft + np.array(*[c_x])
        coo = self.wcs.pixel_to_world(*pos)
        logger.info("Shift (x, y) : (%d, %d) pix / Old (RA,DEC) (%.7f, %.7f) deg / New (RA, DEC) : (%.7f, %.7f) deg" % (*shft, self.coo.ra.value, self.coo.dec.value,
                                                                                                                                    coo.ra.value, coo.dec.value))

        return coo

    def match_catalog(self, sources, it):

        min_sep = self.params["gaia.min_separation"]
        if it > 0: min_sep /= 2.

        ndx_o = []
        ndx_g = []

        gaia_tbl = self.catalog.data

        for i, src in enumerate(sources):
            d = np.sqrt(((src["RA"]-gaia_tbl["RA_ICRS"])*np.cos(np.deg2rad(src["DEC"])))**2+(src["DEC"]-gaia_tbl["DE_ICRS"])**2)
            match_o = np.where((d < min_sep/60./60.))[0]
            if len(match_o) == 1:
                ndx_o.append(i)
                ndx_g.append(match_o[0])

        logger.info("# matched sources : %d (%d/%d)" % (len(ndx_o), len(ndx_o), len(sources)))
        sources = sources[ndx_o]

        return rfn.append_fields(sources, ['GRA', 'GDEC'], [gaia_tbl["RA_ICRS"][ndx_g], gaia_tbl["DE_ICRS"][ndx_g]])

class NIXFrame(Frame):

    def __init__(self, frame, params):
        logger.info("Processing : %s" % frame.filename)
        self.frame = frame
        self.params = params
        self.hdul = self.frame.open()
        self.patch_wcs_if_needed()
        self.setup_gaia()
    
    # This is useless since the recipes like to edit RA and DEC...
    def patch_wcs_if_needed(self):
        pass
    """
        setups = {  "13mas-JHK" : {"crpix1" : 1064.9, "crpix2" : 1099.1, "sign" : 1, "pxscl" : 13},
                    "27mas-JHK" : {"crpix1" : 958.0, "crpix2": 1065.9, "sign" : -1, "pxscl" : 27},
                    "13mas-LM" : {"crpix1" : 1064.9, "crpix2" : 1099.1, "sign" : 1, "pxscl" : 13},}

        if self.hdul[1].header["CTYPE1"] == "PIXEL":
            setup = setups[self.hdul[0].header["ESO INS2 NXCW NAME"]]
            self.hdul[1].header["CTYPE1"] = "RA---TAN"
            self.hdul[1].header["CTYPE2"] = "DEC--TAN"
            self.hdul[1].header["CUNIT1"] = "deg"
            self.hdul[1].header["CUNIT2"] = "deg"
            self.hdul[1].header["CRPIX1"] = setup["crpix1"]
            self.hdul[1].header["CRPIX2"] = setup["crpix2"]
            #self.hdul[1].header["CRVAL1"] = self.hdul[0].header["RA"]
            #self.hdul[1].header["CRVAL2"] = self.hdul[0].header["DEC"]
            self.hdul[1].header["CRVAL1"] = fits.getheader("/data2/DETDATA_PARANAL/NIX/%s" % ".".join(self.frame.filename.split("/")[-1].split(".")[-2:]))["RA"]
            self.hdul[1].header["CRVAL2"] = fits.getheader("/data2/DETDATA_PARANAL/NIX/%s" % ".".join(self.frame.filename.split("/")[-1].split(".")[-2:]))["DEC"]    
            
            PA = self.hdul[0].header["ESO ADA POSANG"]
            cdu = np.array([[setup["sign"], 0], [0, 1]])
            cd = np.matmul(rotM(PA), cdu)*setup["pxscl"]*1e-3/60/60.
            
            self.hdul[1].header["CD1_1"] = cd[0][0] 
            self.hdul[1].header["CD1_2"] = cd[0][1] 
            self.hdul[1].header["CD2_1"] = cd[1][0] 
            self.hdul[1].header["CD2_2"] = cd[1][1] 
    """

    @property
    def pxscl(self):
        if self.hdul[0].header["ESO INS2 NXCW NAME"] == "27mas-JHK":
            return 27.
        else:
            return 13.
    
    @property
    def PA(self):
        return self.hdul[0].header["ESO ADA POSANG"]

    @property
    def sign(self):
        if self.hdul[0].header["ESO INS2 NXCW NAME"] == "27mas-JHK":
            return -1
        else:
            return 1
 
    @property
    def DIT(self):
        return self.hdul[0].header["ESO DET SEQ1 DIT"]

    def setup_gaia(self):
        header_xt0 = self.hdul[0].header
        header_xt1 = self.hdul[1].header
        self.gaia_prod = GAIAProd(SkyCoord(header_xt1["CRVAL1"]*u.deg, header_xt1["CRVAL2"]*u.deg), self.hdul[0].header["MJD-OBS"], self.params)
        # think about this bit
        self.gaia_prod.fetch_catalog(self.pxscl*2048*1.2e-3*np.sqrt(2)*u.arcsec, 20)
        self.gaia_prod.mock_image(self.PA, self.pxscl, self.pxscl*2048*1.2, self.sign)

    def discard_close(self, sources, it):

        min_sep = self.params["sep.min_separation"]
        if it > 0: min_sep /= 2.

        ndx = []

        for i, src in enumerate(sources):

            d = np.sqrt(((src["RA"]-sources["RA"])/np.cos(np.deg2rad(src["DEC"])))**2+(src["DEC"]-sources["DEC"])**2)
            if np.any((0 < d) & (d < min_sep/60./60.)): ndx.append(i)
        
        return np.delete(sources, ndx)

    def add_ra_dec(self, sources):
    
        sz = len(sources)
        ra = np.zeros(sz)
        dec = np.zeros(sz)
        wcs = WCS(self.hdul[1].header)

        for i in range(sz):
            coo = wcs.pixel_to_world(sources[i]["x"], sources[i]["y"])
            ra[i] = coo.ra.value
            dec[i] = coo.dec.value
    
        return rfn.append_fields(sources, ['RA', 'DEC'], [ra, dec]) 

    def extract_sources(self, it):
        
        frame_cut = self.params["nix.frame_cut"]
        sat_level = self.params["nix.saturation_level"]
        thresh = self.params["sep.thresh"]

        image = self.hdul[1].data.byteswap().newbyteorder()
        err = self.hdul[2].data.byteswap().newbyteorder()
        sources = extract(image, thresh, err=err, deblend_cont=1.0)
        sources = sources[np.where((sources["peak"] < sat_level/self.DIT) &
                   (np.abs(sources["x"]-1024) < frame_cut) & (np.abs(sources["y"]-1024) < frame_cut) &
                   ((sources["x"] > 550) | (np.abs(sources["y"]-375) > 150)) &
                   ((sources["x"] > 950) | (np.abs(sources["y"]-1125) > 175)) & 
                   ((np.abs(sources["x"]-800) > 100) | (np.abs(sources["y"]-700) > 100)))]

        sources = self.add_ra_dec(sources)
        sources = self.discard_close(sources, it)

        self.sources = sources

        logger.info("# extracted sources : %d" % len(sources))

    def solve_wcs(self):

        PA = self.PA
        pxscl = self.pxscl

        for i in range(3):
        
            self.extract_sources(i)
            self.sources = self.gaia_prod.match_catalog(self.sources, i)

            #print(np.stack([self.sources["x"], self.sources["y"], self.sources["GRA"], self.sources["GDEC"]], axis=1))
            
            #print(len(self.sources.data))
            if len(self.sources.data) > 4:
                out = leastsq(wcserr_sf, x0=[self.hdul[1].header["CRVAL1"], 
                                        self.hdul[1].header["CRVAL2"], 
                                        PA, pxscl], args=(self.sources.data, self.sign))

                cdu = np.array([[self.sign, 0], [0, 1]])*out[0][3]*1e-3/60./60
                cd = np.matmul(rotM(out[0][2]), cdu)

                self.hdul[1].header["CRVAL1"] = out[0][0]
                self.hdul[1].header["CRVAL2"] = out[0][1]
                self.hdul[1].header["CRPIX1"] = 1023.5
                self.hdul[1].header["CRPIX2"] = 1023.5
                self.hdul[1].header["CD1_1"] = cd[0][0] 
                self.hdul[1].header["CD1_2"] = cd[0][1] 
                self.hdul[1].header["CD2_1"] = cd[1][0] 
                self.hdul[1].header["CD2_2"] = cd[1][1] 

                err = wcserr_sf(out[0], self.sources.data, self.sign)*60*60*1e3

                logger.info("WCSfit / (RA,DEC) : (%.7f,%.7f), PA : %.3f, PXSCL : %.3f, UNC : %.3f+-%.3f" % tuple([*out[0], np.mean(err), np.std(err)]))

                PA = out[0][2]
                pxscl = out[0][3]

            else:
                logger.warning("Not enough matched sources. Skipping...")
                break

        return out[0][0], out[0][1], PA, pxscl, np.mean(err), np.std(err)

    def process(self):
        coo = self.gaia_prod.relative_offset(self.hdul[1].data)
        self.hdul[1].header["CRVAL1"] = coo.ra.value
        self.hdul[1].header["CRVAL2"] = coo.dec.value
        self.hdul[1].header["CRPIX1"] = 1023.5
        self.hdul[1].header["CRPIX2"] = 1023.5
        res = self.solve_wcs()

        return res

class TestRecipe(esorexplugin.RecipePlugin):

    name = "eris_nix_img_wcs_gaia"

    version = VersionNumber(0, 1, 1)

    synopsis = "ERIS NIX WCS analysis."

    description = "ERIS NIX WCS analysis with cross-cataloging to GAIA dr3"

    author = "Yigit Dallilar"
    email = "ydallilar@mpe.mpg.de"

    #copyright = EsoCopyright("TEST_INSTRUMENT", [2010, 2013, 2014, 2015, 2017])

    ENV_ENABLED=False

    parameters = [
            ValueParameter("gaia.maglim", 20.,
                description="Magnitude cut to GAIA catalog", env_enabled=ENV_ENABLED),
            ValueParameter("gaia.prop_motion", False,
                description="Enable GAIA proper motions (seems to not help whole a lot)", env_enabled=ENV_ENABLED),
            ValueParameter("gaia.fwhm", 0.25,
                description="FWHM of sources in GAIA mock image", env_enabled=ENV_ENABLED),
            ValueParameter("gaia.zeromag", 20.,
                description="Zero magnitude of GAIA mock image (shouldn't affect things)", env_enabled=ENV_ENABLED),
            ValueParameter("gaia.min_separation", 0.3,
                description="Maximum angular distance to allow cross-matching", env_enabled=ENV_ENABLED),
            ValueParameter("sep.thresh", 10.,
                description="Source detection threshold for sep in units of background sigma", env_enabled=ENV_ENABLED),
            ValueParameter("sep.min_separation", 1.,
                description="Ignore sources closer than this angular distance", env_enabled=ENV_ENABLED),
            ValueParameter("nix.saturation_level", 15e3,
                description="Saturation level. This is scaled by 1./DIT as images are in adu/s", env_enabled=ENV_ENABLED),
            ValueParameter("nix.frame_cut", 900.,
                description="This is a simple workaround to avoid frame edges. 1024-nix.frame_cut pixels will be cropped as a frame.", env_enabled=ENV_ENABLED),
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

    def set_frame_group(self, frame):
        if frame.tag == 'RAW':
            frame.group = Frame.GROUP_RAW
        else:
            frame.group = Frame.GROUP_CALIB
        frame.level = Frame.LEVEL_FINAL

    def process(self, frames, *args):

        out_cols = ["RA", "DEC", "PA", "PXSCL", "MEAN_UNC", "STD_UNC"]

        params = dict_params(self.input_parameters)
        FrameStore = []
        res = np.recarray((len(frames),), dtype=[(col, float) for col in out_cols])

        output_frames = []
        for i, frame in enumerate(frames):
            nxframe = NIXFrame(frame, params)
            res[i] = nxframe.process()
            FrameStore.append(nxframe)
            new_frame = Frame("%s" % nxframe.frame.filename.split("/")[-1], "PROD", type = Frame.TYPE_IMAGE)
            output_frames.append(new_frame)
        

        for output, frame in zip(output_frames, FrameStore):
            output.write(frame.hdul, overwrite=True)
        
        hdu = fits.BinTableHDU.from_columns(
                fits.ColDefs([fits.Column(name=col, array=res[col], format="D") for col in out_cols]))
        hdu.header["PXSCL"] = np.mean(res["PXSCL"])
        hdu.header["EPXSCL"] = np.std(res["PXSCL"])
        hdu.header["PA"] = np.mean(res["PA"])
        hdu.header["EPA"] = np.std(res["PA"])

        table = Frame("WCS_fit_result.fits", "PROD", type=Frame.TYPE_TABLE)
        output_frames.append(table)
        output_frames[-1].write(hdu, overwrite=True)

        return output_frames

