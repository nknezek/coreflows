#!/usr/bin/python3
import numpy as np
import coremagmodels as cm
import coreflows as cf
sf = cf.advect.SteadyFlow()
wv = cf.advect.Waves()
anl = cf.analyze
import dill
import os
import sys
if len(sys.argv) > 1:
    delta_ths = [float(a) for a in sys.argv[1:]]
else:
    delta_ths = [5,10,15,20,25]

# Import wave fits
data = dill.load(open('/Users/nknezek/code/coreflows/coreflows/data/wavefits012.p','rb'))
c012 = data['c012']
f012 = data['f012']

# Import Steady Flow Fit
filename = '../coreflows/data/steady_flow_fortran_fit'
th_sf,ph_sf,vth_sf,vph_sf = sf.import_fortran_flow_DH(filename)
vth_sfSH = sf.v2vSH(vth_sf)
vph_sfSH = sf.v2vSH(vph_sf)

# Import magnetic model
magmod = cm.models.Chaos6()
T_start = 2001
T_end = 2016
l_max = 14
Nth = 30

th, ph = magmod.get_thvec_phvec_DH(l_max=l_max)
Nt = (T_end-T_start)*3
T = np.linspace(T_start, T_end, Nt)


# compute magnetic field data over timeframe
B_lmax = 14

th, ph = magmod.get_thvec_phvec_DH(Nth)

Bsh = magmod.get_sht_allT(T, l_max = B_lmax)
B = magmod.B_sht_allT(Bsh, Nth=Nth, l_max=B_lmax)
_, dthB, dphB = magmod.gradB_sht_allT(Bsh, Nth=Nth, l_max=B_lmax)

SVsh = magmod.get_SVsht_allT(T, l_max=B_lmax)
SV = magmod.B_sht_allT(SVsh, Nth=Nth, l_max=B_lmax)
_, dthSV, dphSV = magmod.gradB_sht_allT(SVsh, Nth=Nth, l_max=B_lmax)

SAsh = magmod.get_SAsht_allT(T, l_max=B_lmax)
SA = magmod.B_sht_allT(SAsh, Nth=Nth, l_max=B_lmax)
_, dthSA, dphSA = magmod.gradB_sht_allT(SAsh, Nth=Nth, l_max=B_lmax)


# Compute Residual SV 
steadyflow_lmax = 14
SV_steadyflow = sf.SV_steadyflow_allT(vth_sfSH, vph_sfSH, Bsh, magmodel=magmod, Nth=Nth, B_lmax=B_lmax, v_lmax=steadyflow_lmax)
SV_resid = SV-SV_steadyflow

SA_steadyflow = sf.SA_steadyflow_allT(vth_sfSH, vph_sfSH, SVsh, magmodel=magmod, Nth=Nth, B_lmax=B_lmax, v_lmax=steadyflow_lmax)
SA_resid = SA- SA_steadyflow

## Compute Many Correlations

Nphase = 18
phases = np.linspace(0, 180, Nphase, endpoint=False)

period_min = 3
period_max = 15
Nperiod = (period_max-period_min)+1
periods = np.linspace(period_min, period_max, Nperiod, endpoint=False)

for delta_th in delta_ths:
    print('delta_th={}'.format(delta_th))
    weights_s = cf.hermite.fun((th-90)/delta_th,0)
    for m in range(-12,12):
        print('m={}'.format(m))
        for l in (0,1):
            filename = './correlations/l{}m{}dth{}.m'.format(l,m,delta_th)
            if not os.path.isfile(filename):
                wave_params = (l,m,np.nan,1.,np.nan,delta_th)
                SASV_from_phaseperiod = wv.make_SASV_from_phaseperiod_wave_function(wave_params, T, c012, Nth,
                                                                         B=B, dthB=dthB, dphB=dphB, 
                                                                          SV=SV, dthSV=dthSV, dphSV=dphSV)

                SAcorr, SVcorr = cf.analyze.sweep_SASVcrosscorr(phases, periods, T, SA_resid, SV_resid, SASV_from_phaseperiod, 
                                                                weights=weights_s)
                dill.dump((SAcorr,SVcorr),open(filename, 'wb'))
                