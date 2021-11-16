import os
import numpy as np

from astropy.table import Table
import matplotlib.pyplot as plt

import lsst.daf.persistence as dafPersist
import lsst.afw.geom as afwGeom
import lsst.afw.coord as afwCoord
import lsst.afw.image as afwImage
import lsst.afw.display as afwDisplay
import lsst.geom

from astropy.visualization import ZScaleInterval
zscale = ZScaleInterval()


def cutout_coadd_ra_dec(butler, ra, dec, filter='r', cutoutSideLength=59, datasetType='deepCoadd', **kwargs):
    """
    Produce a cutout from coadd from the given butler at the given RA, Dec in decimal degrees.
    
    Notes
    -----
    Trivial wrapper around 'cutout_coadd_spherepoint'
    
    Parameters
    ----------
    butler: lsst.daf.persistence.Butler
        Servant providing access to a data repository
    ra: float
        Right ascension of the center of the cutout, degrees
    dec: float
        Declination of the center of the cutout, degrees
    filter: string
        Filter of the image to load
        
    Returns
    -------
    MaskedImage
    """
    #radec = afwGeom.SpherePointEndpoint(ra, dec, lsst.geom.degrees)# afwGeom.degrees SpherePoint
    radec = lsst.geom.SpherePoint(ra, dec, lsst.geom.degrees)
    return cutout_coadd_spherepoint(butler, radec, filter=filter, datasetType=datasetType, cutoutSideLength=cutoutSideLength)
    

def cutout_coadd_spherepoint(butler, radec, filter='r', datasetType='deepCoadd',
                                  skymap=None, cutoutSideLength=59, **kwargs):
    """
    Produce a cutout from a coadd at the given afw SpherePoint radec position.
    
    Parameters
    ----------
    butler: lsst.daf.persistence.Butler
        Servant providing access to a data repository
    radec: lsst.afw.geom.SpherePoint 
        Coordinates of the center of the cutout.
    filter: string 
        Filter of the image to load
    datasetType: string ['deepCoadd']  
        Which type of coadd to load.  Doesn't support 'calexp'
    skymap: lsst.afw.skyMap.SkyMap [optional] 
        Pass in to avoid the Butler read.  Useful if you have lots of them.
    cutoutSideLength: float [optional] 
        Side of the cutout region in pixels.
    
    Returns
    -------
    MaskedImage
    """
    cutoutSize = lsst.geom.ExtentI(cutoutSideLength, cutoutSideLength)#afwGeom

    if skymap is None:
        skymap = butler.get("%s_skyMap" % datasetType)
    
    # Look up the tract, patch for the RA, Dec
    tractInfo = skymap.findTract(radec)
    patchInfo = tractInfo.findPatch(radec)
    xy = lsst.geom.PointI(tractInfo.getWcs().skyToPixel(radec))#afwGeom
    bbox = lsst.geom.BoxI(xy - cutoutSize//2, cutoutSize)#afwGeom

    coaddId = {'tract': tractInfo.getId(), 'patch': "%d,%d" % patchInfo.getIndex(), 'filter': filter}
    
    cutout_image = butler.get(datasetType+'_sub', bbox=bbox, immediate=True, dataId=coaddId)
    
    return cutout_image


def make_cutout_image(butler, ra, dec, filter='r', vmin=None, vmax=None, label=None,
                      show=True, saveplot=False, savefits=False,
                      datasetType='deepCoadd'):
    """
    Generate and optionally display and save a postage stamp for a given RA, Dec.
    
    Parameters
    ----------
    butler: lsst.daf.persistence.Butler
        Servant providing access to a data repository
    ra: float
        Right ascension of the center of the cutout, degrees
    dec: float
        Declination of the center of the cutout, degrees
    filter: string 
        Filter of the image to load
    Returns
    -------
    MaskedImage

    Notes
    -----
    Uses matplotlib to generate stamps.  Saves FITS file if requested.
    """

    cutout_image = cutout_coadd_ra_dec(butler, ra, dec, filter=filter, datasetType='deepCoadd')
    if savefits:
        if isinstance(savefits, str):
            filename = savefits
        else:
            filename = 'postage-stamp.fits'
        cutout_image.writeFits(filename)
    
    radec = lsst.geom.SpherePoint(ra, dec, lsst.geom.degrees) #afwGeom
    xy = cutout_image.getWcs().skyToPixel(radec)
    
    #if vmin is None or vmax is None:
    #    vmin, vmax = zscale.get_limits(cutout_image.image.array)

    plt.imshow(cutout_image.image.array, vmin=vmin, vmax=vmax,  origin='lower')#cmap='binary_r',
    plt.colorbar()
    plt.scatter(xy.getX() - cutout_image.getX0(), xy.getY() - cutout_image.getY0(),
                color='none', edgecolor='red', marker='o', s=200)
    if label is not None:
        plt.title(label)
    if saveplot:
        if isinstance(saveplot, str):
            filename = saveplot
        else:
            filename = 'postage-stamp.png'
        plt.savefig(filename)
    if show:
        plt.show()

    return cutout_image

