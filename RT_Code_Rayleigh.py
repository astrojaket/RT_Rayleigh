##################################################################
###J.Taylor email:jaketaylorwork@yahoo.co.uk 8/16/2016         ###
###This is a RT code containing an MCMC and Rayleigh scattering###
###we find the temperature of the planet and the radius if the ###
###planet using the MCMC, mod() contains the model. The formulae###
###used in this code can all be found in Le Cavelier Des Etangs ###
###et al. 2008 paper, we solve for the altitude and then we can ###
###use Rp = R_planet + z where Rp is the effective radius which ###
###is what is obsereved as the radius of the planet will vary   ###
###depending on what absorbing molecules/particles are present  ###
###################################################################






import numpy as np
import emcee
import matplotlib.pyplot as plt
from lmfit import minimize, Parameters, report_fit, Minimizer

vals = np.loadtxt('waspdata.csv',delimiter=',')
x = vals[:,0] #wavelengths of data
y = vals[:,1] #rp/rs of data
err = vals[:,2] #errors in data
x_err = vals[:,3]

def mod(teff,r_planet):
	wave = x
	wave_0 = 750
	xsec_0 = 2.52E-28 #cm**2
	#calculate the cross section of the rayleigh scattering
	#for the wavelength range specifed
	#r_planet = 1.224*69911*1000 #meters
	tau_eq = 0.56 #La Cavelier et al 2008
	mmw = 2 * 1.67E-27 #assuming solar abundances
	g = (6.67E-11 * 0.503 * 1.898E27)/(r_planet**2)
	k = 1.38E-23
	#teff = 1145 #value from N.Nikolov 2014
	r_star = 0.87 * 695700 * 1000 #radius of star WASP 6
	h_scale = (k*teff)/(mmw*g) #scale height

	ind = -4
	h_f = 1

	xsec = np.array(xsec_0*((wave/wave_0)**-4)) #cm**2
	#xsec_haze1 = np.array(xsec_h*((wave/wave_h)**ind)) *0.0001
	xsec_conv = xsec * 0.0001 #m2 unit conversion
	pres_0 = np.logspace(1,-2,10)
	pres_conv = pres_0 * 1E5 #pascals
	#print pres_conv[0]

	term1 = ((2*np.pi*r_planet)/(k*teff*mmw*g))**0.5 #this is the path length
	term2 = h_f*pres_conv[0]*xsec_conv #this is mole_frac * reference pressure * cross section
	zed = h_scale * np.log(term1 * (term2/0.56)) #effective altitude

	#print zed
	zed_red = zed/h_scale #reduced altitude wrt scale height; can ignore this was a test
	r_eff = r_planet + zed

	return r_eff/r_star

'''use an lmfit to constrain prior values'''
def residual(params,x, y, err):
	teff = params['temperature'].value
	r_planet = params['r_planet'].value
 	model = mod(teff,r_planet)#telling the likelihood what the model is##


	return (model-y)/err
###use the data to estimate an initial gradient and intercept###

params = Parameters()
params.add('temperature', value=1100.,min=0.,max=3000.)
params.add('r_planet', value=85571064.,min=0.,max=(85571064*100.))


'''this minimises the values, it changes the params input in residual
function but keeps the args values the same each time, it will
vary the values set by params until it reaches a minimum'''
out = minimize(residual, params, args=(x,y, err))

fit = [out.params['temperature'].value,out.params['r_planet'].value]
fit_err = [out.params['temperature'].stderr,out.params['r_planet'].stderr]

report_fit(out)
#this plots how good the LMfit is
plt.plot(x,mod(out.params['temperature'].value,out.params['r_planet'].value),'b',label='Model')
plt.plot(x,y,'ko',label='Data')
plt.errorbar(x, y, xerr= x_err,yerr=err,ls='None',mec='k',ecolor='k')
plt.xlabel('Wavelength nm')
plt.ylabel('Rp/Rs')
plt.title('Generated LMfit plot')
plt.legend()

#plt.ylabel('z/H')
#plt.ylabel('$\delta$F/F')
plt.show()


#print("LM fit[Temperature],[R_planet],[Ind],[Haze_f],[H_f]:",fit)

'''define a prior, returns -inf when reachs the limit, set to zero if constraining'''
#based on the physical knowledge of the parameter we want to minimize
def lnprior(p):
	teff,r_planet = p
	if 0 < teff < 3000. and 0. < r_planet < 8557106400.:
		return 0.0
	else:
		return -np.inf


#used to define the minimisation function
#its  conditional probabilty with respect
def lnlike(p,x,y,err): #assume a gaussian distribution## #func used to define mini func#,guides walkers to build markov chain
	teff,r_planet = p
	model = mod(teff,r_planet)
	chi2 = -0.5*(np.sum(((y - model)/err)**2)) ##using a gaussian to minimize##
	return chi2


def lnprob(p,x,y,err): #now we create the probability function,###
	lp = lnprior(p)
	if not np.isfinite(lp):
		return -np.inf
	return lp + lnlike(p,x,y,err)

ndim = 2
nwalkers = 2*ndim #initiate numvber of dimensions and walkers, play with this
nlinks = 1E5
nlinks_burnin = 1000

pos = [fit + fit_err*np.random.randn(ndim) for i in range(nwalkers)]	#find a start position

#sampler solves for the evidence (division integral)
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(x,y, err))

pos,prob,state = sampler.run_mcmc(pos, nlinks_burnin) #500 iterations#

'''we are taking the transpose as we want to see the paths of the walkers
and how they try and converge to a value'''
fig, axes = plt.subplots(2, 1, sharex=True)
axes[0].plot(sampler.chain[:, :, 0].T, color="k", alpha=0.4)
#axes[0].set_ylim(np.min(sampler.chain[:,:,0].T)*1e-2,np.max(sampler.chain[:,:,0].T)*1e-2)
axes[0].axhline(fit[0], color='blue', lw=1)
axes[0].set_ylabel("Temperature")
axes[0].set_xlabel("Number of interations")
axes[0].set_title("Trace plot for the Rayleigh model")


axes[1].plot(sampler.chain[:, :, 1].T, color="k", alpha=0.4)
axes[1].axhline(fit[1], color='blue', lw=1)
axes[1].set_ylabel("R_planet")
axes[1].set_xlabel("Number of interations")
#
# axes[2].plot(sampler.chain[:, :, 2].T, color="k", alpha=0.4)
# axes[2].axhline(fit[2], color='blue', lw=1)
# axes[2].set_ylabel("Haze_f")
# axes[2].set_xlabel("Number of interations")
#
# axes[3].plot(sampler.chain[:, :, 3].T, color="k", alpha=0.4)
# axes[3].axhline(fit[3], color='blue', lw=1)
# axes[3].set_ylabel("xsec_h")
# axes[3].set_xlabel("Number of interations")
#
# axes[4].plot(sampler.chain[:, :, 4].T, color="k", alpha=0.4)
# axes[4].axhline(fit[4], color='blue', lw=1)
# axes[4].set_ylabel("wave_f")
# axes[4].set_xlabel("Number of interations")
# #plt.show()

'''reset the sampler so that we can get rid of our
burn in and use the new position found'''
sampler.reset()
pos,prob,state = sampler.run_mcmc(pos, nlinks) #this is the same code as before but from new pos


'''plot results'''
import corner
corner.corner(sampler.flatchain,truths=fit,labels=['Temperature', 'R_planet'],quantiles=[0.16, 0.5, 0.84],
                       show_titles=True, title_kwargs={"fontsize": 8})
#plt.savefig('cornerplot.png')
plt.show()
teff_mcmc,r_planet_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                             zip(*np.percentile(sampler.flatchain, [16, 50, 84],
                                                axis=0)))


def gelman_rubin(x, return_var=False):
    """ Returns estimate of R for a set of traces.
    The Gelman-Rubin diagnostic tests for lack of convergence by comparing
    the variance between multiple chains to the variance within each chain.
    If convergence has been achieved, the between-chain and within-chain
    variances should be identical. To be most effective in detecting evidence
    for nonconvergence, each chain should have been initialized to starting
    values that are dispersed relative to the target distribution.
    Parameters
    ----------
    x : array-like
      An array containing the 2 or more traces of a stochastic parameter. That is, an array of dimension m x n x k, where m is the number of traces, n the number of samples, and k the dimension of the stochastic.

    return_var : bool
      Flag for returning the marginal posterior variance instead of R-hat (defaults of False).
    Returns
    -------
    Rhat : float
      Return the potential scale reduction factor, :math:`\hat{R}`
    Notes
    -----
    The diagnostic is computed by:
      .. math:: \hat{R} = \frac{\hat{V}}{W}
    where :math:`W` is the within-chain variance and :math:`\hat{V}` is
    the posterior variance estimate for the pooled traces. This is the
    potential scale reduction factor, which converges to unity when each
    of the traces is a sample from the target posterior. Values greater
    than one indicate that one or more chains have not yet converged.
    References
    ----------
    Brooks and Gelman (1998)
    Gelman and Rubin (1992)"""

    if np.shape(x) < (2,):
        raise ValueError(
            'Gelman-Rubin diagnostic requires multiple chains of the same length.')

    try:
        m, n = np.shape(x)
    except ValueError:
        return [gelman_rubin(np.transpose(y)) for y in np.transpose(x)]

    # Calculate between-chain variance
    B_over_n = np.sum((np.mean(x, 1) - np.mean(x)) ** 2) / (m - 1)

    # Calculate within-chain variances
    W = np.sum(
        [(x[i] - xbar) ** 2 for i,
         xbar in enumerate(np.mean(x,
                                   1))]) / (m * (n - 1))

    # (over) estimate of variance
    s2 = W * (n - 1) / n + B_over_n

    if return_var:
        return s2

    # Pooled posterior variance estimate
    V = s2 + B_over_n / m
    # Calculate PSRF
    R = V / W

    return R

print("""MCMC result:
    Temperature = {0[0]} +{0[1]} -{0[2]}
    R_planet = {1[0]} +{1[1]} -{1[2]}
""".format(teff_mcmc, r_planet_mcmc))
print('Gelman Rubin statistic output:',  gelman_rubin(sampler.chain,return_var=False))

'''plot the graph of the data vs model found using MCMC'''


plt.plot(x,mod(teff_mcmc[0],r_planet_mcmc[0]) ,'b',label='Model')
plt.plot(x,y,'ko',label='Data')
plt.errorbar(x, y, xerr=x_err,yerr=err,ls='None',mec='k',ecolor='k')
plt.xlabel('Wavelength nm')
plt.ylabel('Rp/Rs')
plt.legend()
plt.title('Generated MCMC plot')
#plt.ylabel('z/H')
#plt.ylabel('$\delta$F/F')
plt.show()
