import math
import numpy as np
from scipy.optimize import fsolve
from scipy.stats import norm
import datetime
import pandas as pd
import matplotlib.pyplot as plt

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






def treeUnder(start,T,Nt,sigma=None,u=None,d=None):

    dt = T/Nt
    Ns = Nt+1
    
    if u is None:
        u = np.exp(sigma * np.sqrt(dt))
        d = np.exp(-sigma * np.sqrt(dt))
        
    grid = np.empty((Ns,Nt+1))
    grid[:] = np.nan
    
    tree = pd.DataFrame(grid)
    
    for t in tree.columns:
        for s in range(0,t+1):
            tree.loc[s,t] = start * (d**s * u**(t-s))

    treeinfo = pd.Series({'u':u,'d':d,'Nt':Nt,'dt':dt}).T
            
    return tree, treeinfo




def treeAsset(funPayoff, treeUnder,treeInfo, Z=None, pstar=None, style='european'):
    treeV = pd.DataFrame(np.nan,index= list(range(int(treeInfo.Nt+1))),columns= list(range(int(treeInfo.Nt+1))))
    
    if style=='american':
        treeExer = treeV.copy()
    
    for t in reversed(treeV.columns):
        if t ==treeV.columns[-1]:
            for s in treeV.index:
                treeV.loc[s,t] = funPayoff(treeUnder.loc[s,t]) 
                if style=='american':
                    if treeV.loc[s,t]>0:
                        treeExer.loc[s,t] = True
                    else:
                        treeExer.loc[s,t] = False
                    
        else:
            probvec = [pstar[t-1],1-pstar[t-1]]

            for s in treeV.index[:-1]:        
                treeV.loc[s,t] = Z[t-1] * treeV.loc[[s,s+1],t+1] @ probvec
                
                if style=='american':
                    exerV = funPayoff(treeUnder.loc[s,t])
                    if exerV > treeV.loc[s,t]:
                        treeExer.loc[s,t] = True
                        treeV.loc[s,t] = exerV
                    else:
                        treeExer.loc[s,t] = False

    if style=='american':
        return treeV, treeExer
    else:
        return treeV
    
    
    
    
def bs_delta_to_strike(under,delta,sigma,T,isCall=True,r=0):
    
    if isCall:
        phi = 1
    else:
        phi = -1
        if delta > 0:
            delta *= -1
        
    strike = under * np.exp(-phi * norm.ppf(phi*delta) * sigma * np.sqrt(T) + .5*sigma**2*T)
    
    return strike
    

    
    
    
    
    
    
    
    
    
### Auxiliary Functions for Handling Nasdaq Skew Data
def load_vol_surface(LOADFILE,SHEET):

    info = pd.read_excel(LOADFILE,sheet_name='descriptions').set_index('specs')
    labels = info.columns

    if type(SHEET) == int or type(SHEET) == float:
        lab = labels[SHEET]
    else:
        lab = SHEET
        
    raw = pd.read_excel(LOADFILE,sheet_name=lab).set_index('Date')

    ts = raw.loc[:,['Future Price','Expiration Future','Expiration Option']]
    surf = raw.drop(columns=ts.columns)

    indPuts = surf.columns.str.contains('P')
    indCalls = surf.columns.str.contains('C')

    calls = surf[surf.columns[indCalls]]
    puts = surf[surf.columns[indPuts]]

    return ts, puts



def get_notable_dates(opts, ts, maxdiff=False):

    if maxdiff==True:
        dtgrid = pd.DataFrame([opts.diff().abs().idxmax()[0], ts[['Future Price']].diff().abs().idxmax()[0]],columns=['notable date'],index=['max curve shift','max underlying shift'])
    else:
        dtgrid = pd.DataFrame([opts.diff().abs().idxmax()[0], ts[['Future Price']].pct_change().abs().idxmax()[0]],columns=['notable date'],index=['max curve shift','max underlying shift'])
    for row in dtgrid.index:
        dtgrid.loc[row,'day before'] = opts.loc[:dtgrid.loc[row, 'notable date'],:].index[-2]
    dtgrid = dtgrid.iloc[:,::-1].T
    
    return dtgrid
    
    
    
    
    
    
def get_strikes_from_vol_moneyness(TYPE,opts,ts):

    if TYPE == 'call':
            phi = 1
            isCall = True
    else:
        phi =-1
        isCall = False

    deltas = pd.DataFrame(np.array([float(col[1:3])/100 for col in opts.columns]) * phi, index=opts.columns,columns = ['delta'])

    strikes = pd.DataFrame(index=opts.index, columns=opts.columns, dtype=float)
    for t in opts.index:
        T = ts.loc[t,'Expiration Option']
        for col in deltas.index:
            strikes.loc[t,col] = bs_delta_to_strike(under = ts.loc[t,'Future Price'], delta=deltas.loc[col,'delta'], sigma=opts.loc[t,col], T=T, isCall=isCall)
            
            
    return strikes






def graph_vol_surface_as_strikes(dtgrid,opts,strikes,ts,label):

    fig, ax = plt.subplots(2,1,figsize=(10,10))

    for j, dt in enumerate(dtgrid.columns):

        colorgrid = ['b','r','g']

        for i, tstep in enumerate(dtgrid[dt]):
            tstep = tstep.strftime('%Y-%m-%d')
            plotdata = pd.concat([opts.loc[tstep,:],strikes.loc[tstep,:]],axis=1)
            plotdata.columns = [tstep,'strike']
            plotdata.set_index('strike',inplace=True)
            plotdata.plot(ax=ax[j],color=colorgrid[i]);

            ax[j].axvline(x=ts.loc[tstep,'Future Price'],color=colorgrid[i],linestyle='--')

            if j==0:        
                ax[j].set_title(f'Curve shock: {label}')
            elif j==1:
                ax[j].set_title(f'Underlying shock: {label}')

            if label.split(' ')[-2] == 'ED':
                ax[j].set_xlim(xmin=0,xmax=.08)
            
            plt.tight_layout()

