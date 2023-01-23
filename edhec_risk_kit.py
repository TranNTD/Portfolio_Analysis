import pandas as pd
import scipy.stats
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
from matplotlib import pyplot as plt

def drawdown(returns_series: pd.Series):
    """
    takes a times series od asset returns
    Computes and returns a DF that contains:
    the wealth index
    the previous peaks
    percent drawdowns
    """
    wealth_index = 1000*(1+returns_series).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks)/previous_peaks
    return pd.DataFrame({
        "Wealth": wealth_index,
        "Peaks": previous_peaks,
        "Drawdown": drawdowns
    })

def get_ffme_returns():
        """
        Load the Fama-French DataSet for the returns of the Top and Bottoms Deciles by MarketCap
        """
        me_m =pd.read_csv("data/Portfolios_Formed_on_ME_monthly_EW.csv",
            header=0, index_col=0, na_values=-99.99)
        rets = me_m[['Lo 10', 'Hi 10']]
        rets.columns = ['SmallCap','LargeCap']
        rets = rets/100
        rets.index = pd.to_datetime(rets.index, format = "%Y%m").to_period('M')
        return rets

def get_hfi_returns():
        hfi = pd.read_csv("data/edhec-hedgefundindices.csv", header=0, index_col=0, parse_dates=True)
        hfi = hfi/100
        hfi.index = hfi.index.to_period('M')
        return hfi
    
def get_ind_returns():
    '''
    Load and format the EDHEC Hedge Fund Index Returns
    '''
    ind=pd.read_csv("data/ind49_m_vw_rets.csv", header=0, index_col=0, parse_dates=True)/100
    ind.index = pd.to_datetime(ind.index, format = "%Y%m").to_period("M")
    ind.columns = ind.columns.str.strip()
    return ind
    
def semideviation(r):
    '''
    Returns the Semi Deviation aka negative semideviation of r.
    r mus be a Series or DataFrame
    '''
    is_negative = r < 0
    return r[is_negative].std(ddof=0)

def var_historic(r, level = 5):
    '''
    VaR Histories
    '''
    if isinstance(r, pd.DataFrame):
        return r.aggregate(var_historic, level=level)
    elif isinstance(r, pd.Series):
        return -np.percentile(r,level)
    else:
        raise TypeError('Expected r to be Series or DataFrame')


def skewness(r):
    demeaned_r = r-r.mean()
    # Use the population standard deviation, so set degree of freedom = 0
    sigma_r = r.std(ddof=0)
    exp=(demeaned_r**3).mean()
    return exp/sigma_r**3

def kurtosis(r):
    demeaned_r = r-r.mean()
    # Use the population standard deviation, so set degree of freedom = 0
    sigma_r = r.std(ddof=0)
    exp=(demeaned_r**4).mean()
    return exp/sigma_r**4

def is_normal(r, level = 0.01):
    '''
    Apllies the Jarque-Bera test to determine if a Series is normal or not.
    Test is applied at the 1% level by default.
    Return True if the hypothesis of normality is accepted, False otherwise
    '''
    statistic, p_value = scipy.stats.jarque_bera(r)
    return p_value > level

def VaR_gaussian(r,level=5, modified = False):
    '''
    Returns the Parametric Gaussian VaR of a Series or dataFrame
    if 'modified' is True, than the modified VaR is returned, using the Cornish-Fisher modification
    '''
    z = norm.ppf(level/100)
    if modified:
        S = skewness(r)
        K = kurtosis(r)
        z = z + (S/6)*(z**2-1) + (1/24)*(z**3-3*z)*(K-3) - ((S**2)/36)*(2*z**3-5*z)
    return -(r.mean() + z*r.std(ddof=0))

def cvar_historic(r, level=5):
    '''
    Compute the Conditional VaR of Series or DataFrame
    '''
    if isinstance(r, pd.Series):
        is_beyond = r <= -var_historic(r,level=level)
        return -r[is_beyond].mean()
    elif isinstance(r, pd.DataFrame):
        return r.aggregate(cvar_historic, level=level)
    else:
        raise TypeError("Expected r to be a Series or DataFrame") 

def annualize_returns(r, periods_per_year):
    compounded_growth = (1+r).prod()
    n_periods = r.shape[0]
    return compounded_growth**(periods_per_year/n_periods)-1
    
def annualize_vol(r, periods_per_year):
    """
    Annualizes the vol of a set of returns
    We should infer the periods per year
    """
    return r.std()*(periods_per_year**0.5)

def sharpe_ratio(r, riskfree_rate, periods_per_year):
    #convert the annual riskfree rate to per period
    rf_per_period = (1+riskfree_rate)**(1/periods_per_year)-1
    excess_returns = r - rf_per_period
    ann_excess_return = annualize_returns(excess_returns, periods_per_year)
    annual_vol = annualize_vol(r, periods_per_year)
    return ann_excess_return/annual_vol
    
def portfolio_return(weights, returns):
    """
    Weights -> Returns
    """
    return weights.T @ returns

def portfolio_vol(weights, covmat):
    """
    weights -> vol
    """
    
    return (weights.T @ covmat @ weights)**0.5

def plot_ef2(n_points, expected_r, cov, style = '.-'):
    """
    Plot the 2-Asset efficient Frontier
    """
    if expected_r.shape[0] !=2:
        raise ValueError("plot_ef2 can only plot 2-asset frontiers")
    weights =[np.array([w,1-w]) for w in np.linspace(0,1,n_points) ]
    rets = [portfolio_return(w, expected_r) for w in weights]
    vols = [portfolio_vol(w, cov) for w in weights]
    ef = pd.DataFrame({
        "Returns":rets,
        "Volatility":vols
    })
    return ef.plot.line(x="Volatility", y="Returns", style = style)


def minimize_vol(target_r, er, cov):
    '''
    Generate the weights of the target Return
    '''
    n = er.shape[0]
    init_guests = np.repeat(1/n,n)
    bounds = ((0.0, 1.0),)*n # n copy from (0,1) tuple
    return_is_target = {
        'type':'eq',
        'args':(er,),
        'fun': lambda weights, er: target_r-portfolio_return(weights,er) #=0 if target met
    }
    weights_sum_to_1= {
        'type':'eq',
        'fun': lambda weights: np.sum(weights)-1
    }
    results = minimize(portfolio_vol, init_guests, args = (cov,), method="SLSQP",
                          options={'disp':False},
                          constraints=(return_is_target, weights_sum_to_1),
                           bounds=bounds
                          )
    return results.x

def optimal_weights(n_points, er, cov):
    '''
     Generate a list of weights to run the optimizer on to minimize the volatility
    '''
    target_return =np.linspace(er.min(),er.max(),n_points) 
    weights = [minimize_vol(target_r,er,cov) for target_r in target_return]
    return weights
def gmv(cov):
    '''
    Returns the weight of the Global Minimum Vol Portfolio given the covmat
    '''
    n = cov.shape[0]
    return msr(0, np.repeat(1,n), cov)
    
def plot_ef(n_points, expected_r, cov, style = ".-",show_cml = False, riskfree_rate=0, show_ew=False, show_gmv = False):
    """
    Plot the N-Asset efficient Frontier
    """
    weights = optimal_weights(n_points, expected_r, cov)
    rets = [portfolio_return(w, expected_r) for w in weights]
    vols = [portfolio_vol(w, cov) for w in weights]
    ef = pd.DataFrame({
        "Returns":rets,
        "Volatility":vols
    })
    
    ax = ef.plot.line(x="Volatility", y="Returns", title = "{}-Asset Efficient Frontier".format(n_points), style = style)
    ax.set_ylabel('Returns')
    ax.set_xlabel('Volatility')
 
    if show_gmv:
        w_gmv = gmv(cov)
        r_gmv = portfolio_return(w_gmv, expected_r) 
        vol_gmv = portfolio_vol(w_gmv, cov)
        # display EW
        ax.plot(vol_gmv, r_gmv, color ='midnightblue', marker='o', markersize=12)
        plt.text(vol_gmv, r_gmv, "GMV-Portfolio")
    
    if show_ew:
        n = expected_r.shape[0]
        w_ew = np.repeat(1/n,n)
        r_ew = portfolio_return(w_ew, expected_r) 
        vol_ew = portfolio_vol(w_ew, cov)
        # display EW
        ax.plot(vol_ew, r_ew, color ='goldenrod', marker='o', markersize=12)
        plt.text(vol_ew, r_ew, "EW-Portfolio")
        
    if show_cml:
        ax.set_xlim(left = 0)
        w_msr = msr(riskfree_rate, expected_r, cov)
        r_msr = portfolio_return(w_msr, expected_r)
        vol_msr = portfolio_vol(w_msr, cov)
        # Add CML
        cml_x = [0, vol_msr]
        cml_y = [riskfree_rate, r_msr]
        ax.plot(cml_x,cml_y, color="green", marker="o", linestyle="dashed", markersize=12, linewidth=2)
        plt.text(cml_x[1], cml_y[1], '({:.2f},{:.2f})'.format(cml_x[1],cml_y[1]))
        
    return ax

def msr(riskfree_rate, er, cov):
    '''
    RiskfreeRate + Expected Return + Cov -> Weights of msr
    Optimize the sharpe ratio = minimize the neg sharpe ratio
    '''
    n = er.shape[0]
    init_guests = np.repeat(1/n,n)
    bounds = ((0.0, 1.0),)*n # n copy from (0,1) tuple
    weights_sum_to_1= {
        'type':'eq',
        'fun': lambda weights: np.sum(weights)-1
    }
    def neg_sharpe_ratio(weights, riskfree_rate, er, cov):
        '''
        Return the negative of the sharpe ratio
        '''
        r = portfolio_return(weights, er)
        vol = portfolio_vol(weights, cov)
        return -(r-riskfree_rate)/vol
        
        
    results = minimize(neg_sharpe_ratio, init_guests, args = (riskfree_rate, er, cov,), method="SLSQP",
                          options={'disp':False},
                          constraints=(weights_sum_to_1),
                           bounds=bounds
                          )
    return results.x
    
                      
    
    
    