""" Simple object to hold DLAs"""

class Lls():
    def __init__(self, central_wavelength, col_density=0, id = None,s2n=None):
        self.central_wavelength = central_wavelength    # observed wavelength (1+zDLA)*1215.6701
        self.col_density = col_density                  # log10 column density
        self.id = id                                    # str, the id of DLA
        self.s2n=s2n