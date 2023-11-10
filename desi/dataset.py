import numpy as np
import scipy.signal as signal
from lls_cnnmodel.desi.training_sets import split_sightline_into_samples
from lls_cnnmodel.desi.preprocess import label_sightline
from lls_cnnmodel.spectra_utils import get_lam_data
from lls_cnnmodel.training_set import select_samples_50p_pos_neg
from lls_cnnmodel.desi import defs

REST_RANGE = defs.REST_RANGE
kernel = defs.kernel
smooth_kernel = defs.smooth_kernel
best_v = defs.best_v


def make_datasets(sightlines, kernel=kernel, REST_RANGE=REST_RANGE, v=best_v['all'], output='datasets.npy',
                  validate=True):
    """
    Generate training set or validation set for DESI.
    
    Parameters:
    -----------------------------------------------
    sightlines: list of 'dla_cnn.data_model.Sightline' object, the sightlines should be preprocessed.
    validate: bool
    
    Returns
    -----------------------------------------------
    dataset:dict, the training set contains flux and 3 labels, the validation set contains flux, lam, 3 labels and DLAs' data.
    
    """
    dataset = {}
    for sightline in sightlines:
        wavelength_dlas = [dla.central_wavelength for dla in sightline.dlas]
        coldensity_dlas = [dla.col_density for dla in sightline.dlas]
        label_sightline(sightline, kernel=kernel, REST_RANGE=REST_RANGE)
        data_split = split_sightline_into_samples(sightline, REST_RANGE=REST_RANGE, kernel=kernel, v=v)
        if validate:
            flux = np.vstack([data_split[0]])
            labels_classifier = np.hstack([data_split[1]])
            labels_offset = np.hstack([data_split[2]])
            col_density = np.hstack([data_split[3]])
            lam = np.vstack([data_split[4]])  # no need lam
            dataset[sightline.id] = {'FLUX': flux, 'labels_classifier': labels_classifier,
                                     'labels_offset': labels_offset, 'col_density': col_density,
                                     'wavelength_dlas': wavelength_dlas, 'coldensity_dlas': coldensity_dlas}
        else:
            sample_masks = select_samples_50p_pos_neg(sightline, kernel=kernel)
            if sample_masks != []:
                flux = np.vstack([data_split[0][m] for m in sample_masks])
                labels_classifier = np.hstack([data_split[1][m] for m in sample_masks])
                labels_offset = np.hstack([data_split[2][m] for m in sample_masks])
                col_density = np.hstack([data_split[3][m] for m in sample_masks])
                dataset[sightline.id] = {'FLUX': flux, 'labels_classifier': labels_classifier,
                                         'labels_offset': labels_offset, 'col_density': col_density}
    np.save(output, dataset)
    return dataset

