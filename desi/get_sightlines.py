from lls_cnnmodel.desi.preprocess import estimate_s2n, normalize, rebin
from lls_cnnmodel.desi.defs import best_v

import numpy as np
import os
from os.path import join
from astropy.table import Table, vstack

from DesiMockpu import DesiMock


def get_sightlines(path='', output='pre_sightlines.npy'):
    """
    Read fits file and preprocess the sightlines
    
    Return
    ---------
    sightlines:list of `lls_cnnmodel.data_model.sightline.Sightline` object
    
    """
    sightlines = []

    # get folder list
    item1 = os.listdir(path)

    for k in item1:
        #get subfolder list
        if (0 < int(k)) & (int(k) < 31):
        #if k==17:
            tt = path + '/' + str(k)
            item = os.listdir(tt)
            for j in item:

                file_path = path + '/' + str(k) + '/' + str(j)
                s = os.listdir(file_path)
                # for k in glob.glob(os.path.join(dd, '*.fits')):
                if len(s) > 2:
                    file_path = path + '/' + str(k) + '/' + str(j)
                    spectra = join(file_path, "spectra-16-%s.fits" % j)
                    truth = join(file_path, "truth-16-%s.fits" % j)
                    zbest = join(file_path, "zbest-16-%s.fits" % j)
                    specs = DesiMock()
                    specs.read_fits_file(spectra, truth, zbest)
                    keys = list(specs.data.keys())
                    for jj in keys:
                        sightline = specs.get_sightline(jj, camera='all', rebin=False,
                                                    normalize=True)
                        # calculate S/N
                        sightline.s2n = estimate_s2n(sightline)
                        if (sightline.z_qso >= 2.33) & (sightline.s2n >=3):  # apply filtering 2.33
                            # only use blue band data
                            sightline.flux = sightline.flux[0:sightline.split_point_br]
                            sightline.error = sightline.error[0:sightline.split_point_br]
                            sightline.loglam = sightline.loglam[0:sightline.split_point_br]
                            rebin(sightline, best_v['b'])
                            sightlines.append(sightline)

    np.save(output, sightlines)
    return sightlines
