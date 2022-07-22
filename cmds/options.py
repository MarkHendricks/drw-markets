import math
import numpy as np
from scipy.optimize import fsolve
import datetime
import pandas as pd

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

def bs_price(under=None,strike=None,T=None,rf=None,vol=None,option='call'):
    d1,d2 = bs_normargs(under=under,strike=strike,T=T,rf=rf,vol=vol) 
    
    if option=='put':
        return np.exp(-rf*T)*strike * normal_cdf(-d2) - under * normal_cdf(-d1)
    else:
        return under * normal_cdf(d1) - np.exp(-rf*T)*strike * normal_cdf(d2)


def bs_rho(under=None,strike=None,T=None,rf=None,vol=None):

    d1,d2 = bs_normargs(under=under,strike=strike,T=T,rf=rf,vol=vol)
    return normal_cdf(d2) * strike * T * np.exp(-rf*T)



def bs_impvol(under=None,strike=None,T=None,rf=None,option='call',opt_price=None,volGuess=.25,showflag=False):
    func = lambda ivol: (opt_price-bs_price(vol=ivol,under=under,strike=strike,T=T,rf=rf,option=option))**2
    xstar, analytics, flag, msg = fsolve(func, volGuess, full_output=True)
    
    if showflag:
        return xstar, msg
    else:
        return xstar
    
    
def to_maturity(expiration=None, current_date=None):
    return (pd.to_datetime(expiration) - pd.to_datetime(current_date)).total_seconds()/(24*60*60)/365


def filter_stale_quotes(opt_chain):
    LDATE =opt_chain.sort_values('lastTradeDate')['lastTradeDate'].iloc[-1]
    mask = list()

    for idx in opt_chain.index:
        dt = opt_chain.loc[idx,'lastTradeDate']
        if (dt - LDATE).total_seconds()/3600 > -24:
            mask.append(idx)
    
    return mask

def clean_options(calls_raw,puts_raw):
    idx = filter_stale_quotes(calls_raw)
    calls = calls_raw.loc[idx,:]
    idx = filter_stale_quotes(puts_raw)
    puts = puts_raw.loc[idx,:]

    calls = calls[calls['volume'] > calls['volume'].quantile(.75)].set_index('contractSymbol')
    puts = puts[puts['volume'] > puts['volume'].quantile(.75)].set_index('contractSymbol')
    
    calls['lastTradeDate'] = calls['lastTradeDate'].dt.tz_localize(None)
    puts['lastTradeDate'] = puts['lastTradeDate'].dt.tz_localize(None)
    
    return calls, puts