import math
import numpy as np

def normal_cdf(x):
    return(1 + math.erf(x/np.sqrt(2)))/2

def normal_pdf(x):
    return np.exp(-x**2/2) / np.sqrt(2*np.pi)

def bs_normargs(under=None,strike=None,T=None,rf=None,vol=None):
    d1 = (np.log(under/strike) + (rf + .5 * vol**2)*T ) / (vol * np.sqrt(T))
    d2 = d1 - vol * np.sqrt(T)
    return [d1,d2]

def bs_delta(under=None,strike=None,T=None,rf=None,vol=None):
    d1 = bs_normargs(under=under,strike=strike,T=T,rf=rf,vol=vol)[0]

    return normal_cdf(d1)         

def bs_theta(under=None,strike=None,T=None,rf=None,vol=None):
        d1,d2 = bs_normargs(under=under,strike=strike,T=T,rf=rf,vol=vol) 
        
        temp = (- under * normal_pdf(d1) * vol)/(2*np.sqrt(T)) - rf * strike * np.exp(-rf*T) * normal_cdf(d2)
        return temp

def bs_gamma(under=None,strike=None,T=None,rf=None,vol=None):
    
    d1 = bs_normargs(under=under,strike=strike,T=T,rf=rf,vol=vol)[0]
    return normal_pdf(d1) / (under * vol * np.sqrt(T))


def bs_vega(under=None,strike=None,T=None,rf=None,vol=None):

    d1 = bs_normargs(under=under,strike=strike,T=T,rf=rf,vol=vol)[0]
    return normal_pdf(d1) * (under * np.sqrt(T))

def bs_price(under=None,strike=None,T=None,rf=None,vol=None):
    d1,d2 = bs_normargs(under=under,strike=strike,T=T,rf=rf,vol=vol) 
    
    return under * normal_cdf(d1) - np.exp(-rf*T)*strike * normal_cdf(d2)


def bs_rho(under=None,strike=None,T=None,rf=None,vol=None):

    d1,d2 = bs_normargs(under=under,strike=strike,T=T,rf=rf,vol=vol)
    return normal_cdf(d2) * strike * T * np.exp(-rf*T)

