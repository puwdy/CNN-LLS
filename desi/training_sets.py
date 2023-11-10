import multiprocessing
from multiprocessing import Pool
import itertools
import numpy as np
from lls_cnnmodel.Timer import Timer
from lls_cnnmodel.spectra_utils import get_lam_data
from lls_cnnmodel.training_set import select_samples_50p_pos_neg
from lls_cnnmodel.desi.defs import REST_RANGE,kernel,best_v

def pad_sightline(sightline, lam, lam_rest, ix_dla_range,kernelrangepx,v=best_v['all']):
    c = 2.9979246e8
    dlnlambda = np.log(1+v/c)
    #pad left side
    if np.nonzero(ix_dla_range)[0][0]<kernelrangepx:
        pixel_num_left=kernelrangepx-np.nonzero(ix_dla_range)[0][0]
        pad_lam_left= lam[0]*np.exp(dlnlambda*np.array(range(-pixel_num_left,0)))
        pad_value_left = np.mean(sightline.flux[0:50])
    else:
        pixel_num_left=0
        pad_lam_left=[]
        pad_value_left=[] 
    #pad right side
    if np.nonzero(ix_dla_range)[0][-1]>len(lam)-kernelrangepx:
        pixel_num_right=kernelrangepx-(len(lam)-np.nonzero(ix_dla_range)[0][-1])
        pad_lam_right= lam[0]*np.exp(dlnlambda*np.array(range(len(lam),len(lam)+pixel_num_right)))
        pad_value_right = np.mean(sightline.flux[-50:])
    else:
        pixel_num_right=0
        pad_lam_right=[]
        pad_value_right=[]
    flux_padded = np.hstack((pad_lam_left*0+pad_value_left, sightline.flux,pad_lam_right*0+pad_value_right))
    lam_padded = np.hstack((pad_lam_left,lam,pad_lam_right))
    return flux_padded,lam_padded,pixel_num_left
    
    
def split_sightline_into_samples(sightline, REST_RANGE=REST_RANGE, kernel=kernel, v=best_v['all']):
    """
    Split the sightline into a series of snippets, each with length kernel

    Parameters
    ----------
    sightline: dla_cnn.data_model.Sightline
    REST_RANGE: list
    kernel: int, optional

    Returns
    -------

    """
    lam, lam_rest, ix_dla_range = get_lam_data(sightline.loglam, sightline.z_qso, REST_RANGE)
    kernelrangepx = int(kernel/2) #200
    #samplerangepx = int(kernel*pos_sample_kernel_percent/2) #60
    #padding the sightline:
    flux_padded,lam_padded,pixel_num_left=pad_sightline(sightline,lam,lam_rest,ix_dla_range,kernelrangepx,v=v)
     
    #ix_dlas = [(np.abs(lam[ix_dla_range]-dla.central_wavelength).argmin()) for dla in sightline.dlas]
    #coldensity_dlas = [dla.col_density for dla in sightline.dlas]       # column densities matching ix_dlas

    # FLUXES - Produce a 1748x400 matrix of flux values
    #fluxes_matrix = np.vstack(map(lambda x:x[0][x[1]-kernelrangepx:x[1]+kernelrangepx],zip(itertools.repeat(sightline.flux), np.nonzero(ix_dla_range)[0])))
    fluxes_matrix = np.vstack(map(lambda x:x[0][x[1]-kernelrangepx:x[1]+kernelrangepx],zip(itertools.repeat(flux_padded), np.nonzero(ix_dla_range)[0]+pixel_num_left)))
    lam_matrix = np.vstack(map(lambda x:x[0][x[1]-kernelrangepx:x[1]+kernelrangepx],zip(itertools.repeat(lam_padded), np.nonzero(ix_dla_range)[0]+pixel_num_left)))
    #using cut will lose side information,so we use padding instead of cutting 
    #fluxes_matrix = np.vstack(map(lambda x:x[0][x[1]-kernelrangepx:x[1]+kernelrangepx],zip(itertools.repeat(sightline.flux), np.nonzero(ix_dla_range)[0][cut])))
    #lam_matrix = np.vstack(map(lambda x:x[0][x[1]-kernelrangepx:x[1]+kernelrangepx],zip(itertools.repeat(lam), np.nonzero(ix_dla_range)[0][cut])))
    #the wavelength and flux array we input:
    input_lam=lam_padded[np.nonzero(ix_dla_range)[0]+pixel_num_left]
    input_flux=flux_padded[np.nonzero(ix_dla_range)[0]+pixel_num_left]
    # Return
    return fluxes_matrix, sightline.classification, sightline.offsets, sightline.column_density,lam_matrix,input_lam,input_flux
    #return fluxes_matrix, sightline.classification, sightline.offsets, sightline.column_density
