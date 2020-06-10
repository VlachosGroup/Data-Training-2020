#%% Import necessary packages

import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn import linear_model 
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_validate, LeaveOneOut, train_test_split
from scipy.stats import norm

matplotlib.rcParams['font.size'] = 20
matplotlib.rcParams['axes.linewidth'] = 1.5
matplotlib.rcParams['xtick.major.size'] = 8
matplotlib.rcParams['xtick.major.width'] = 2
matplotlib.rcParams['ytick.major.size'] = 8
matplotlib.rcParams['ytick.major.width'] = 2
matplotlib.rcParams['figure.dpi'] = 300.


#%% helper functions
def parity_plot(X, y, model, model_name): 
    
    '''
    #plot parity plot
    '''
    y_predict_all = model.predict(X)
    #y_predict_all = predict_y(pi_nonzero, intercept, J_nonzero)
    
    plt.figure(figsize=(4,4))
    
    fig, ax = plt.subplots()
    ax.scatter(y, y_predict_all, s=60, facecolors='none', edgecolors='r')
    
    plt.xlabel("Actual value")
    plt.ylabel("Model prediction")
    
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]
    
    # now plot both limits against eachother
    ax.plot(lims, lims, 'k--', alpha=0.75, zorder=0)
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    plt.tight_layout()

def error_distribution(yobj, ypred, model_name):
    
    '''
    Plot the error distribution
    return the standard deviation of the error distribution
    '''
    fig, ax = plt.subplots(figsize=(4,4))
    ax.hist(yobj - ypred,density=1, alpha=0.5, color='steelblue')
    mu = 0
    sigma = np.std(yobj - ypred)
    x_resid = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
    ax.plot(x_resid, norm.pdf(x_resid, mu, sigma), color='r')
    plt.title(r'{}, $\sigma$-{:.2}'.format(model_name, sigma))
    
    return sigma




#%% Plot functions
'''
Bar plot comparing performance
'''

regression_method = [ 'PLS', 'PCR', 'PCR Geometric', 'SPCR Geometric']
scores_mx = np.array([  scores_PLS, scores_pc2,  scores_pcg_second,  scores_spc2])
means_test = np.array(scores_mx[:,2])
r2s = np.array([  r2_PLS,  r2_pc2,  r2_pcg_second, r2_spc2])

x_pos = np.arange(len(regression_method))
base_line = 0

opacity = 0.9
bar_width = 0.2


fig, ax1 = plt.subplots(figsize=(8,6))
ax2 = ax1.twinx()
rects2 = ax1.bar(x_pos, means_test - base_line, bar_width, #yerr=std_test,  
                alpha = opacity, color='salmon',
                label='Test')
rects3 = ax2.bar(x_pos+bar_width, r2s - base_line, bar_width, #yerr=std_test,  
                alpha = opacity, color='lightgreen',
                label='r2')
#plt.ylim([-1,18])
ax1.set_xticks(x_pos+bar_width/2)
#ax1.set_xticklabels(regression_method, rotation=0)
ax1.set_xticklabels(regression_method, rotation=10)
ax1.set_xlabel('Predictive Models')
#plt.legend(loc= 'best', frameon=False)

ax1.set_ylabel('Testing RMSE (eV)', color = 'k')
ax1.set_ylim([0, 1])
ax1.tick_params('y', colors='k')

ax2.set_ylabel('$R^2$',color = 'k')
ax2.set_ylim([0, 1])
ax2.tick_params('y', colors='k')
plt.legend(loc= 'upper left', frameon=False)

plt.tight_layout()
#fig.savefig(os.path.join(output_dir, model_name + '_performance.png'