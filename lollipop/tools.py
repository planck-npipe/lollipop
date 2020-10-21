import numpy as np
import mpfit as mp
from numpy import linalg
import astropy.io.fits as fits


def chi2tolik( chi2):
    chi2 = asarray( chi2)
    chi2[isnan(chi2)] = inf
    return( exp(-0.5*( chi2 - nanmin(chi2))))


def ctr_level(histo2d, lvl):
    """
    Extract the contours for the 2d plots
    """
    
    h = histo2d.flatten()*1.
    h.sort()
    cum_h = cumsum(h[::-1])
    cum_h /= cum_h[-1]
    
    alvl = searchsorted(cum_h, lvl)
    clist = h[-alvl]
    
    return clist

def confidence_lvl( x, lik, lvl):
    import numpy as np
    from scipy.interpolate import interp1d
    from scipy.optimize import bisect,brentq

    dx = (max(x)-min(x))/(len(x)-1)
    ix = np.arange( min(x)+dx, max(x), dx/1000.)
    f = interp1d( x, lik, kind='quadratic')
    dist = f(ix)
    
    iML = np.argmax( dist)
    
    dist.sort()
    cum = np.cumsum( dist[::-1])
    cum /= cum[-1]
    
    alvl = np.searchsorted( cum, lvl)
    
    f2 = interp1d( x, lik-dist[-alvl], kind='quadratic')
#    clm = bisect( f2, min(x), ix[iML])# - ix[iML]
#    clp = bisect( f2, ix[iML], max(x))# - ix[iML]
    try:
        clm = brentq( f2, min(x), ix[iML])# - ix[iML]
    except ValueError:
        clm = -np.inf
    try:
        clp = brentq( f2, ix[iML], max(x))# - ix[iML]
    except ValueError:
        clp = +np.inf
    return( clm,ix[iML],clp)


def getci( c, symmetrical=False):
    if symmetrical:
        return(c[1], mean([abs(c[0]-c[1]), abs(c[2]-c[1])]))
    else:
        return(c[1], c[0]-c[1], c[2]-c[1])



def FeldmanCousins( xmin, sigma, CL=95):
    import os
    from scipy.interpolate import interp1d

    fcdata = loadtxt( "/sps/planck/Users/tristram/Planck/FeldmanCousins.data").T
    listCL = [68,90,95,99]

    if not(CL in listCL):
        raise ValueError('CL not in the list')

    clmin = interp1d( fcdata[0], fcdata[listCL.index(CL)*2+1])
    clmax = interp1d( fcdata[0], fcdata[listCL.index(CL)*2+2])
    
    x0=xmin/sigma
    if (x0<min(fcdata[0])) or (x0>max(fcdata[0])):
        return( NaN)
    uplim = clmax(x0)
    
    return( uplim*sigma)


def posterior2D( xval, yval, L, labels=[" "," "], xlim=None, ylim=None, cmap="Blues"):
    from matplotlib import pyplot as plt
    
    extent = [xval.min(),xval.max(),yval.min(),yval.max()]
    if xlim is not None:
        extent[0:2] = [xlim[0],xlim[1]]
    if ylim is not None:
        extent[2:] = [ylim[0],ylim[1]]
    extent = tuple(extent)
    
    #2D plot
    ax=plt.subplot( 2, 2, 3)
    lvls=append(list(ctr_level( L, [0.68,0.95])[::-1]),L.max())
    plt.contourf( xval, yval, L, levels=lvls, colors=None, cmap=plt.cm.get_cmap(cmap))
    plt.xlabel( labels[0])
    plt.ylabel( labels[1])
    plt.xlim( extent[0:2])
    plt.ylim( extent[2:])
    ax.locator_params(tight=True, nbins=6)
    
    #yval
    ax=plt.subplot( 2, 2, 4)
    plt.plot( yval, sum( L,1)/sum(L), 'k')
    ax.yaxis.set_visible(False)
    ax.locator_params(tight=True, nbins=6)
    plt.xlim( extent[2:])
    plt.xlabel( labels[1])
    
    #xval
    ax=plt.subplot( 2, 2, 1)
    plt.plot( xval, sum( L, 0)/sum(L), 'k')
    ax.locator_params(tight=True, nbins=6)
    ax.yaxis.set_visible(False)
    plt.xlim( extent[0:2])
    
    #text
    ax=plt.subplot( 2, 2, 1)
    ci = array(getci(confidence_lvl(xval,sum(L,0),0.68),symmetrical=True))
    plt.text( 1.25, 0.5, "%s = $%5.3f \pm %5.3f$" % (labels[0], ci[0], mean(abs(ci[1:]))), transform=ax.transAxes)
    ci = array(getci(confidence_lvl(yval,sum(L,1),0.68)))
    plt.text( 1.25, 0.6, "%s = $%5.3f \pm %5.3f$" % (labels[1], ci[0], mean(abs(ci[1:]))), transform=ax.transAxes)


