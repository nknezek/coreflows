import pyshtools as sht
import numpy as np
import coreflows as cm

class Advect(cm.SphereHarmBase):
    def __init__(self):
        pass

    def v2vSH(self, v):
        return sht.shtools.SHExpandDH(v, norm=1, sampling=2, csphase=1)

    def v2vSH_allT(self, v_t):
        lm = (v_t.shape[1]-2)//2
        vSH_t = np.empty((v_t.shape[0],2,lm+1,lm+1))
        for i in range(v_t.shape[0]):
            vSH_t[i,:,:,:] = sht.shtools.SHExpandDH(v_t[i,:,:], norm=1, sampling=2, csphase=1)
        return vSH_t

    def vSH2v_allT(self, vSH_t, l_max=None, Nth=None):
        if Nth is None and l_max is None:
            l_max = self._get_lmax(vSH_t[0])
            Nth = l_max * 2 + 2
            lm = l_max
        elif Nth is None:
            lm = l_max
            Nth = lm*2+2
        else:
            lm = (Nth-2)/2
        v_t = np.empty((len(vSH_t), Nth, Nth*2))
        for i,vSH in enumerate(vSH_t):
            SH = self._convert_SHin(vSH, l_max=l_max)
            v_t[i,:,:] = sht.shtools.MakeGridDH(SH, norm=1, sampling=2, csphase=1, lmax_calc=l_max, lmax=lm)
        return v_t

    def vSH2v(self, vSH, l_max=None, Nth=None):
        if Nth is None and l_max is None:
            l_max = self._get_lmax(vSH)
            Nth = l_max * 2 + 2
            lm = l_max
        elif Nth is None:
            lm = l_max
        else:
            lm = (Nth-2)/2
        SH = self._convert_SHin(vSH, l_max=l_max)
        return sht.shtools.MakeGridDH(SH, norm=1, sampling=2, csphase=1, lmax_calc=l_max, lmax=lm)

    def gradient_vSH(self, vSH, l_max=None, Nth=None):
        '''
        calculates the horizontal gradient of a velocity field given in spherical harmonics [km/yr]

        :param self:
        :param vSH: velocity field spherical harmonics [km/yr]
        :param l_max:
        :param Nth:
        :return: gradients of the field in [ km / yr-km ]
        '''
        if Nth is None and l_max is None:
            l_max = self._get_lmax(vSH)
            Nth = l_max*2+2
            lm = l_max
        if Nth is None:
            lm = l_max
        else:
            lm = (Nth - 2) / 2
        SH = self._convert_SHin(vSH, l_max=l_max)
        out = sht.shtools.MakeGravGridDH(SH, 3480, 3480, sampling=2, normal_gravity=0, lmax_calc=l_max, lmax=lm)
        dth_v = out[1]
        dph_v = out[2]
        return dth_v, dph_v

    def gradients_v_allT(self, v_t, Nth=None, l_max=None):
        dth_t = np.empty_like(v_t)
        dph_t = np.empty_like(v_t)
        for i, t in enumerate(range(v_t.shape[0])):
            dSH = self.v2vSH(v_t[i, :, :])
            dth_t[i, :, :], dph_t[i, :, :] = self.gradient_vSH(dSH, l_max=l_max, Nth=Nth)
        return dth_t, dph_t

    def dth_v(self, vSH, l_max=14, Nth=None):
        '''
        computes the latitudinal gradient of a velocity field given with spherical harmonics

        :param self:
        :param vSH: spherical harmonics of a velocity field with units [km/yr]
        :param l_max:
        :param Nth:
        :return: gradient of velocity field in units of [ km / (yr-km) ]
        '''

        if Nth is None:
            lm = l_max
        else:
            lm = (Nth-2)/2
        SH = self._convert_SHin(vSH, l_max=l_max)
        _,dth_v, _, _ = sht.shtools.MakeGravGridDH(SH, 3480, 3480, sampling=2, normal_gravity=0, lmax_calc=l_max, lmax=lm)
        return dth_v

    def dph_v(self, vSH, l_max=14, Nth=None):
        '''
        computes the longitudinal gradient of a velocity field given with spherical harmonics

        :param self:
        :param vSH: spherical harmonics of a velocity field with units [km/yr]
        :param l_max:
        :param Nth:
        :return: gradient of velocity field in units of [ km / (yr-km) ]
        '''
        if Nth is None:
            lm = l_max
        else:
            lm = (Nth-2)/2
        SH = self._convert_SHin(vSH, l_max=l_max)
        _,_,dph_v,_ = sht.shtools.MakeGravGridDH(SH, 3480, 3480, sampling=2, normal_gravity=0, lmax_calc=l_max, lmax=lm)
        return dph_v

    def advSV(self, vthSH, vphSH, BSH, l_max=14, Nth=None, B_lmax = None, v_lmax=None, B_in_vSHform=False):
        if v_lmax is None:
            v_lmax = l_max
        if B_lmax is None:
            B_lmax = l_max
        vth = self.vSH2v(vthSH, l_max=v_lmax, Nth=Nth)
        vph = self.vSH2v(vphSH, l_max=v_lmax, Nth=Nth)
        if B_in_vSHform:
            dthB, dphB = self.gradient_vSH(BSH, l_max=B_lmax, Nth=Nth)
        else:
            drB, dthB, dphB = self.gradB_sht(BSH, l_max=B_lmax, Nth=Nth)
        return - vth*dthB - vph*dphB

    def divSV(self, vthSH,vphSH,BSH, l_max=14, Nth=None, B_lmax = None, v_lmax=None, B_in_vSHform=False):
        if v_lmax is None:
            v_lmax = l_max
        if B_lmax is None:
            B_lmax = l_max
        dthvth = self.dth_v(vthSH, l_max=v_lmax, Nth=Nth)
        dphvph = self.dph_v(vphSH, l_max=v_lmax, Nth=Nth)
        if B_in_vSHform:
            B = self.vSH2v(BSH, l_max=B_lmax, Nth=Nth)
        else:
            B = self.B_sht(BSH, l_max=B_lmax, Nth=Nth)
        return -B*(dthvth + dphvph)

    def SV_from_flow(self, vthSH, vphSH, BSH, l_max=14, Nth=None, B_lmax = None, v_lmax=None, B_in_vSHform=False):
        '''
        Takes flow and field and computes secular variation from flow advection of background field

        :param vthSH: latitudinal flow [km/yr]
        :param vphSH: longitudinal flow [km/yr]
        :param BSH: field [nT]
        :param l_max:
        :param Nth:
        :param B_lmax:
        :param v_lmax:
        :return: secular variation [nT/yr]
        '''
        return self.advSV(vthSH,vphSH,BSH, l_max=l_max, Nth=Nth, B_lmax = B_lmax, v_lmax=v_lmax, B_in_vSHform=B_in_vSHform) \
                + self.divSV(vthSH,vphSH,BSH, l_max=l_max, Nth=Nth, B_lmax = B_lmax, v_lmax=v_lmax, B_in_vSHform=B_in_vSHform)

    def SV_steadyflow_allT(self, vthSH, vphSH, BSH_t, Nth, B_lmax=14, v_lmax=14):
        SVsteadyflow_t = np.empty((len(BSH_t), Nth, Nth * 2))
        for i, bSH in enumerate(BSH_t):
            SVsteadyflow_t[i, :, :] = self.SV_from_flow(vthSH, vphSH, bSH, B_lmax=B_lmax, v_lmax=v_lmax, Nth=Nth)
        return SVsteadyflow_t

    def SA_from_flow_accel(self, athSH, aphSH, BSH, l_max=14, Nth=None, B_lmax = None, v_lmax=None, B_in_vSHform=False):
        '''
        Takes flow and field and computes secular acceleration from flow advection of background field

        :param vthSH: latitudinal flow [km/yr]
        :param vphSH: longitudinal flow [km/yr]
        :param BSH: field [nT]
        :param l_max:
        :param Nth:
        :param B_lmax:
        :param v_lmax:
        :return: secular variation [nT/yr]
        '''
        return self.SV_from_flow(athSH, aphSH, BSH, l_max=l_max, Nth=Nth, B_lmax=B_lmax, v_lmax=v_lmax, B_in_vSHform=B_in_vSHform)

    def SA_from_flow_SV(self, vthSH, vphSH, svSH, l_max=14, Nth=None, B_lmax = None, v_lmax=None, B_in_vSHform=False):
        '''
        Takes flow and field and computes secular acceleration from flow advection of secular variation

        :param vthSH: latitudinal flow [km/yr]
        :param vphSH: longitudinal flow [km/yr]
        :param BSH: field [nT]
        :param l_max:
        :param Nth:
        :param B_lmax:
        :param v_lmax:
        :return: secular variation [nT/yr]
        '''
        return self.SV_from_flow(vthSH, vphSH, svSH, l_max=l_max, Nth=Nth, B_lmax=B_lmax, v_lmax=v_lmax, B_in_vSHform=B_in_vSHform)

    def SA_steadyflow_allT(self, vthSH, vphSH, SVsh_t, Nth, B_lmax=14, v_lmax=14):
        ''' compute the secular acceleration of steady flow for all T

        :param vthSH:
        :param vphSH:
        :param SVsh_t:
        :param Nth:
        :param B_lmax:
        :param v_lmax:
        :return:
        '''
        SAsteadyflow_t = np.empty((len(SVsh_t), Nth, Nth * 2))
        for i, SVsh in enumerate(SVsh_t):
            SAsteadyflow_t[i, :, :] = self.SA_from_flow_SV(vthSH, vphSH, SVsh, B_lmax=B_lmax, v_lmax=v_lmax, Nth=Nth)
        return SAsteadyflow_t

    def tp2v(self, torSH, polSH, l_max=14):
        vtht, vpht = self.t2v(torSH, l_max=l_max)
        vthp, vphp = self.p2v(polSH, l_max=l_max)
        vth = vtht + vthp
        vph = vpht + vphp
        return vth, vph

    def t2v(self, torSH, l_max=14):
        z = self._convert_SHin(torSH, l_max=l_max)
        _, dth, dph, _ = sht.shtools.MakeGravGridDH(z, 3480, 3480, lmax=l_max, sampling=2)
        vtht = -dph
        vpht = dth
        return vtht, vpht

    def p2v(self, polSH, l_max=14):
        z = self._convert_SHin(polSH, l_max=l_max)
        _,_,_,_,dtr,dpr = sht.shtools.MakeGravGradGridDH(z, 3480,3480, sampling=2)
        vthp = dtr
        vphp = dpr
        return vthp, vphp

    def import_fortran_flow_DH(self, filename):
        th = []
        ph = []
        ang = []
        mag = []
        with open(filename, 'rb') as velfile:
            for line in velfile:
                ln_cln = line.strip().decode('UTF-8').split()
                ph.append(float(ln_cln[0]))
                th.append(float(ln_cln[1]))
                ang.append(float(ln_cln[2]))
                mag.append(float(ln_cln[3]))
        th = np.array(th)
        ph = np.array(ph)
        ph = (ph + 180.) % 360. - 180.
        ang = np.array(ang)
        mag = np.array(mag)
        vph = mag * np.sin(ang * np.pi / 180)
        vth = mag * np.cos(ang * np.pi / 180)
        keys = np.lexsort((ph, th))
        n = int((len(th) / 2) ** 0.5)
        vth_trans = np.reshape(vth[keys], (n, 2 * n))[::-1, ::]
        vph_trans = np.reshape(vph[keys], (n, 2 * n))[::-1, ::]
        th_trans = np.reshape(th[keys], (n, 2 * n))[::-1, ::]
        ph_trans = np.reshape(ph[keys], (n, 2 * n))[::-1, ::]
        return th_trans, ph_trans, -vth_trans, vph_trans

    def import_fortran_flow_tpSH(self, filename):
        '''
        Perhaps Doesn't work correctly yet.

        Returns
        -------

        '''
        raw = []
        with open(filename, 'rb') as velfile:
            velfile.readline()
            velfile.readline()
            for line in velfile:
                ln_cln = line.strip().decode('UTF-8').split()
                for num in ln_cln:
                    raw.append(float(num))
        n = len(raw) // 2
        l_max = int((n + 1) ** 0.5) - 1
        toroidal = raw[:n]
        poloidal = raw[n:]
        tcoeffs = self._convert_g_raw_to_shtarray(toroidal, l_max=l_max)
        tSH = sht.SHCoeffs.from_array(tcoeffs, normalization='schmidt', csphase=-1)
        pcoeffs = self._convert_g_raw_to_shtarray(poloidal, l_max=l_max)
        pSH = sht.SHCoeffs.from_array(pcoeffs, normalization='schmidt', csphase=-1)
        return tSH, pSH

    # def fit_l_over_t_using_fft(self, T, data, Nfft=10, scale_mult=1):
    #     dfft = np.fft.fft(data)
    #     fft_mag = np.random.normal(loc=np.zeros(Nfft), scale=np.abs(dfft[:Nfft])*scale_mult)
    #     fft_phase = np.random.uniform(size=Nfft)
    #     ffts = fft_mag * np.exp(1j * 2 * np.pi * fft_phase)
    #     ys = np.fft.ifft(ffts, n=len(T))
    #     return ys.real
    #
    # def simulate_coeffs_fft(self, T, shcoeffs_t, Nfft=10, scale_mult=1, scale_l=0, l_max=14):
    #     l_arr = min(shcoeffs_t.shape[2], l_max)
    #     shcoeffs_sim = np.zeros((shcoeffs_t.shape[0], shcoeffs_t.shape[1], l_arr, l_arr))
    #     for k in range(shcoeffs_t.shape[1]):
    #         for l in range(l_arr):
    #             for m in range(l_arr):
    #                 shcoeffs_sim[:, k, l, m] = self.fit_l_over_t_using_fft(T, shcoeffs_t[:, k, l, m], Nfft=Nfft, scale_mult=scale_mult*(l**scale_l))
    #     return shcoeffs_sim

class Analyze(cm.SphereHarmBase):
    def __init__(self):
        pass

    def correlation(self, f, g, T, th=None, ph=None, R=3480., thmax=None):
        '''
        Compute correlation of functions f and g with shapes (i_time, i_th, i_ph) where i_ph = 2*i_th

        Parameters
        ----------
        f: dim (N time, N th, N ph)
        g: dim (N time, N th, N ph)
        T: array of time coords
        th: array of theta coords
        dph: spacing of ph coords
        R: radius [m]

        Returns
        -------

        '''

    def _convert_weights(self, f, weights):
        '''converts weights to the correct dimensions for f on a sphere'''
        Nth = f.shape[1]
        dtt = np.pi / Nth
        tt = np.linspace(dtt / 2, np.pi - dtt / 2, Nth)

        if len(weights.shape)==1:
            if weights.shape[0] == 1:
                wt = weights[:, None, None]
            elif weights.shape[0] == f.shape[1]:
                wt = weights[None, :, None]
            elif weights.shape[0] == f.shape[0]:
                wt = weights[:,None, None]
            else:
                raise ValueError('weights wrong shape')
        elif len(weights.shape)==2:
                if weights.shape == f.shape[:2]:
                    wt = weights[:,:,None]
                else:
                    raise ValueError('weights wrong shape')
        else:
            raise ValueError('weights wrong shape')
        return np.sin(tt)[None,:,None]*wt

    def standard_deviation(self, f, weights=np.ones((1)), fwm=None):
        ''' computes the standard deviation on a sphere of f

        :param f:
        :param weights:
        :return:
        '''
        wt = self._convert_weights(f,weights)
        fw = f*wt
        if fwm is None:
            fwm = np.mean(fw)
        return np.sqrt(np.sum((fw - fwm) ** 2))

    def cross_correlation(self, f, g, weights=np.ones((1)), gw=None, fw=None, fwm=None, gwm=None, fwsd=None, gwsd=None, swt=None):
        ''' compute cross-correlation of f and g on a sphere with weight function across theta

        :param f:
        :param g:
        :param weights:
        :return:
        '''
        if swt is None:
            swt = self._convert_weights(f,weights)
        if fw is None:
            fw = f * swt
        if gw is None:
            gw = g * swt
        if fwm is None:
            fwm = np.mean(fw)
        if gwm is None:
            gwm = np.mean(gw)
        if fwsd is None:
            fwsd = self.standard_deviation(fw, fwm=fwm)
        if gwsd is None:
            gwsd = self.standard_deviation(gw, fwm=gwm)
        cross_cov = np.sum((fw - fwm) * (gw - gwm))
        cross_corr = cross_cov / (fwsd * gwsd)
        return cross_corr

    def convolve(self, f, g, T, th=None, ph=None, R=3480., thmax=None, weights=1.):
        '''
        Convolve functions f and g with shapes (i_time, i_th, i_ph) where i_ph = 2*i_th

        Parameters
        ----------
        f: dim (N time, N th, N ph)
        g: dim (N time, N th, N ph)
        T: array of time coords
        th: array of theta coords
        dph: spacing of ph coords
        R: radius [m]
        thmax: maximum latitude off the equator in degrees
        Returns
        -------

        '''
        dt = T[1] - T[0]
        n = f.shape[1]
        if th is None:
            th = np.linspace(0, np.pi, n, endpoint=False)
            dth = np.pi / n
        else:
            dth = th[1] - th[0]

        if thmax is not None:
            min_ind = np.where((90 - thmax) * np.pi / 180 < th)[0][0]
            max_ind = np.where((90 + thmax) * np.pi / 180 > th)[0][-1]
            f = f[:,min_ind:max_ind+1,:]
            g = g[:,min_ind:max_ind+1,:]
            th = th[min_ind:max_ind+1]

        if ph is None:
            dph = 2 * np.pi / (2 * n)
        else:
            dph = ph[1]-ph[0]
        fg = (f * g)
        diff = weights*np.sin(th) * dth * dph * R ** 2 * dt
        conv = np.sum(np.tensordot(fg, diff, axes=(1, 0)))/(4*np.pi*R**2*(T[-1]-T[0]))
        return conv

    def sweep_convolution(self, data, fit_fun, phases, periods, T, th, ph, thmax=None):
        '''
        Sweeps convolution across provided array of phases and periods

        Parameters
        ----------
        data
        fit_fun
        phases
        periods

        Returns
        -------

        '''
        power = np.zeros((len(phases), len(periods)))
        for i, p in enumerate(phases):
            for j, t in enumerate(periods):
                power[i, j] = self.convolve(data, fit_fun(p,t), T, th, ph=ph, thmax=thmax)
        return power

    def rms_region(self, z, lat=None, lon=None, weights=1., R=3480e3, axis=None):
        '''
        find the rms of z in region of thmin - thmax, pmin-pmax measured in degrees latitude and longitude
        z: data of size len(th), len(ph)
        th: location in degrees latitude (evenly spaced)
        ph: location in degrees longitude (evenly spaced)
        '''
        if lat is None:
            Nth = z.shape[0]
            dth = 180/Nth
            lat = np.linspace(-90+dth/2,90-dth/2,Nth)
        else:
            dth = (lat[1] - lat[0]) * np.pi / 180
        if lon is None:
            Nph = z.shape[1]
            dph = 360/Nph
            lon = np.linspace(-180+dph/2,180-dph/2,Nph)
        else:
            dph = (lon[1] - lon[0]) * np.pi / 180
        colat = lat + 90
        pp, tt = np.meshgrid(lon, colat)
        area = np.sum(np.abs(weights) * np.sin(tt * np.pi / 180) * dth * dph * R ** 2, axis=axis)
        return np.sqrt(np.sum(np.abs(z)**2 * weights * np.sin(tt * np.pi / 180) * dth * dph * R ** 2, axis=axis) / area)

    def rms_region_allT(self, z, lat=None, lon=None, weights=1., R=3480e3, axis=None):
        '''find the rms of z in region of thmin - thmax, pmin-pmax measured in degrees latitude and longitude

        z: data of size len(th), len(ph)
        th: location in degrees latitude (evenly spaced)
        ph: location in degrees longitude (evenly spaced)
        '''
        if lat is None:
            Nth = z.shape[1]
            dth = 180/Nth
            lat = np.linspace(-90+dth/2,90-dth/2,Nth)
        else:
            dth = (lat[1] - lat[0]) * np.pi / 180
        if lon is None:
            Nph = z.shape[2]
            dph = 360/Nph
            lon = np.linspace(-180+dph/2,180-dph/2,Nph)
        else:
            dph = (lon[1] - lon[0]) * np.pi / 180
        colat = lat + 90
        pp, tt = np.meshgrid(lon, colat)
        if type(weights) is np.ndarray:
            if len(weights.shape) == 2:
                if weights.shape == z.shape[1:]:
                    wt = weights
                elif weights.shape[0] == z.shape[1] and weights.shape[1] == 1:
                    wt = weights.repeat(z.shape[2], axis=1)
                else:
                    raise TypeError('weight wrong shape')
            elif len(weights.shape) == 1:
                if weights.shape[0] == z.shape[1]:
                    wt = weights[:,np.newaxis].repeat(z.shape[2], axis=1)
                else:
                    raise TypeError('weight wrong shape')
            else:
                raise TypeError('weight wrong shape')
        elif type(weights) is float:
            wt = weights

        area = np.sum(np.abs(wt) * np.sin(tt * np.pi / 180) * dth * dph * R ** 2, axis=axis)
        z_rms = 0
        for i in range(z.shape[0]):
            z_rms += np.sqrt(np.sum(np.abs(z[i,:,:])**2 * wt * np.sin(tt * np.pi / 180) * dth * dph * R ** 2, axis=axis) / area)
        return z_rms/z.shape[0]

    def weighted_mean_region(self, z, lat=None, lon=None, weights=1., R=3480e3, axis=None):
        '''
        find the mean of z in region of thmin - thmax, pmin-pmax measured in degrees latitude and longitude

        z: data of size len(th), len(ph)
        th: location in degrees latitude (evenly spaced)
        ph: location in degrees longitude (evenly spaced)
        '''
        if lat is None:
            Nth = z.shape[0]
            dth = 180/Nth
            lat = np.linspace(-90+dth/2,90-dth/2,Nth)
        else:
            dth = (lat[1] - lat[0]) * np.pi / 180
        if lon is None:
            Nph = z.shape[1]
            dph = 360/Nph
            lon = np.linspace(-180+dph/2,180-dph/2,Nph)
        else:
            dph = (lon[1] - lon[0]) * np.pi / 180
        colat = lat + 90
        pp, tt = np.meshgrid(lon, colat)
        area = np.sum(np.abs(weights) * np.sin(tt * np.pi / 180) * dth * dph * R ** 2, axis=axis)
        return np.sum(z * weights * np.sin(tt * np.pi / 180) * dth * dph * R ** 2, axis=axis) / area

    def weighted_mean_region_allT(self, z, th=None, ph=None, weights=1., R=3480e3, axis=1):
        '''
        computes weighted mean in a region

        :param z: data in nparray with dimension [len(T), len(th), len(ph)]
        :param th: nparray of colatitudes
        :param ph: nparray of longitudes
        :param weights: nparray with dimension [len(th)]
        :param R: radius of spherical surface
        :param axis: axis upon which to perform mean 0:time, 1: colatitude, 3: longitude. (default: 1)
        :return:
        '''
        if th is None:
            Nth = z.shape[1]
            dth = 180/Nth
            th = np.linspace(dth/2,180-dth/2,Nth)
        else:
            dth = (th[1] - th[0]) * np.pi / 180
        if ph is None:
            Nph = z.shape[2]
            dph = 360/Nph
            ph = np.linspace(dph/2,360-dph/2,Nph)
        else:
            dph = (ph[1] - ph[0]) * np.pi / 180
        pp, tt = np.meshgrid(ph, th)
        if len(weights.shape) == 2:
            if weights.shape == z.shape[1:]:
                wt = weights
            elif weights.shape[0] == z.shape[1] and weights.shape[1] == 1:
                wt = weights.repeat(z.shape[2], axis=1)
            else:
                raise TypeError('weight wrong shape')
        elif len(weights.shape) == 1:
            if weights.shape[0] == z.shape[1]:
                wt = weights[:,np.newaxis].repeat(z.shape[2], axis=1)
            else:
                raise TypeError('weight wrong shape')
        else:
            raise TypeError('weight wrong shape')

        if axis>0:
            area_axis = axis - 1
            area = np.sum(np.abs(wt) * np.sin(tt * np.pi / 180) * dth * dph * R ** 2, axis=area_axis)
        else:
            area = np.abs(wt) * np.sin(tt * np.pi / 180) * dth * dph * R ** 2
        return np.sum(z*wt*np.sin(tt*np.pi/180) * dth * dph * R ** 2, axis=axis) / area

    def normal(self, x, mu, sigma):
        return (np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) / (2 * sigma ** 2 * np.pi) ** 0.5)[:, None]

    def compute_frequency_wavenumber(self, y, T, fourier_mult=20):
        '''
        computes freq / wavenumber 2D fft for given data (y), with axes of time [0] and longitude [1].
        fourier_mult determines the sampling rate for the fourier transform
        '''
        Tend = T[-1]
        Tstart = T[0]
        Nph = y.shape[1]
        Nt = y.shape[0]

        Nphf = int(Nph * fourier_mult)
        Tphf = 1 * fourier_mult / Nphf
        Ntf = int(Nt * fourier_mult)
        Ttf = (Tend - Tstart) * fourier_mult / Ntf

        yf = np.fft.fft2(y, s=[Ntf, Nphf])
        yf = np.fft.fftshift(yf, axes=1)

        m = np.linspace(-1 / (2 * Tphf), 1 / (2 * Tphf), Nphf)
        freq = np.linspace(0, 1 / (2 * Ttf), Ntf // 2)
        return m, freq, yf

    def compute_frequency_wavenumber_region(self, z, T, fourier_mult=10, m_min=-10, m_max=10,
                                            period_min=2.5, period_max=32, return_axis_labels=True):
        '''
        take 2dfft of particular m and period region
        '''
        m, freq, zf = self.compute_frequency_wavenumber(z, T, fourier_mult=fourier_mult)
        # find proper data to plot
        ximin = np.where(m > m_min)[0][0]
        ximax = np.where(m < m_max)[0][-1]
        yimax = np.where(freq < 1 / period_min)[0][-1]
        yimin = np.where(freq > 1 / period_max)[0][0]

        # set up axes
        m_plt = m[ximin:ximax]
        freq_plt = freq[yimin:yimax]
        z_plt = np.abs(zf[yimin:yimax, ximin:ximax])

        if return_axis_labels:
            freq_label_loc = np.linspace(freq_plt[0], freq_plt[-1], 10)  # 10 is simply the number of labels on chart
            period_labels = ['{:.1f}'.format(1 / x) for x in freq_label_loc]
            return m_plt, freq_plt, z_plt, freq_label_loc, period_labels
        else:
            return m_plt, freq_plt, z_plt

    def sweep_SASVconv(self, phases, periods, T, SA_t, SV_t, SASV_from_phaseperiod_function, weights=1., normalize=False):
        ''' computes the cross-correlation between observed SA/SV and SA/SV produced by a wave of a series of periods and phases

        :param phases:
        :param periods:
        :param T:
        :param SA_t:
        :param SV_t:
        :param SASV_from_periodphase_function:
        :param latmax:
        :return:
        '''
        SAcorr = np.empty((len(phases), len(periods)))
        SVcorr = np.empty((len(phases), len(periods)))
        if normalize:
            SAautocorr0 = np.sqrt(self.convolve(SA_t, SA_t, T, weights=weights))
            SVautocorr0 = np.sqrt(self.convolve(SV_t, SV_t, T, weights=weights))
        for i, phase in enumerate(phases):
            for j, period in enumerate(periods):
                SAwave_t, SVwave_t = SASV_from_phaseperiod_function(phase, period)
                if normalize:
                    SAautocorr = np.sqrt(self.convolve(SAwave_t, SAwave_t, T, weights=weights)) * SAautocorr0
                    SVautocorr = np.sqrt(self.convolve(SVwave_t, SVwave_t, T, weights=weights)) * SVautocorr0
                    SAcorr[i, j] = self.convolve(SAwave_t, SA_t, T, weights=weights)/SAautocorr
                    SVcorr[i, j] = self.convolve(SVwave_t, SV_t, T, weights=weights)/SVautocorr
                else:
                    SAcorr[i, j] = self.convolve(SAwave_t, SA_t, T, weights=weights)
                    SVcorr[i, j] = self.convolve(SVwave_t, SV_t, T, weights=weights)
            print('finished phase {}/{}'.format(i + 1, len(phases)))
        return SAcorr, SVcorr

    def sweep_SASVcrosscorr(self, phases, periods, T, SA_t, SV_t, SASV_from_phaseperiod_function, weights=np.ones((1)), print_update=True):
        ''' computes the cross-correlation between observed SA/SV and SA/SV produced by a wave of a series of periods and phases

        :param phases:
        :param periods:
        :param T:
        :param SA_t:
        :param SV_t:
        :param SASV_from_periodphase_function:
        :param latmax:
        :return:
        '''
        SAcrosscorr = np.empty((len(phases), len(periods)))
        SVcrosscorr = np.empty((len(phases), len(periods)))

        swt = self._convert_weights(SA_t, weights)
        SAswt = SA_t * swt
        SAswtm = np.mean(SAswt)
        SAswtsd = self.standard_deviation(SAswt, fwm=SAswtm)

        SVswt = SV_t * swt
        SVswtm = np.mean(SVswt)
        SVswtsd = self.standard_deviation(SVswt, fwm=SVswtm)

        for i, phase in enumerate(phases):
            for j, period in enumerate(periods):
                SAwave_t, SVwave_t = SASV_from_phaseperiod_function(phase, period)
                SAcrosscorr[i, j] = self.cross_correlation(SA_t, SAwave_t, swt=swt, fw=SAswt, fwm=SAswtm, fwsd=SAswtsd)
                SVcrosscorr[i, j] = self.cross_correlation(SV_t, SVwave_t, swt=swt, fw=SVswt, fwm=SVswtm, fwsd=SVswtsd)
            if print_update:
                print('\r\t\tfinished phase {}/{}'.format(i + 1, len(phases)), end='')
        return SAcrosscorr, SVcrosscorr

    def sweep_SVcrosscorr(self, phases, periods, T, SV_t, SASV_from_phaseperiod_function, weights=np.ones((1)), print_update=True):
        ''' computes the cross-correlation between observed SV and SV produced by a wave of a series of periods and phases

        :param phases:
        :param periods:
        :param T:
        :param SA_t:
        :param SV_t:
        :param SASV_from_periodphase_function:
        :param latmax:
        :return:
        '''
        SVcrosscorr = np.empty((len(phases), len(periods)))

        swt = self._convert_weights(SV_t, weights)

        SVswt = SV_t * swt
        SVswtm = np.mean(SVswt)
        SVswtsd = self.standard_deviation(SVswt, fwm=SVswtm)

        for i, phase in enumerate(phases):
            for j, period in enumerate(periods):
                _, SVwave_t = SASV_from_phaseperiod_function(phase, period)
                SVcrosscorr[i, j] = self.cross_correlation(SV_t, SVwave_t, swt=swt, fw=SVswt, fwm=SVswtm, fwsd=SVswtsd)
            if print_update:
                print('\r\t\tfinished phase {}/{}'.format(i + 1, len(phases)), end='')
        return SVcrosscorr

    def sweep_amplitudes(self, SA_obs, SA_waves_list, amp_min=0.1, amp_max=5, Namps=20, weights=1.):
        ''' Computes the goodness-of-fit across an array of amplitudes for each wave,

        given the observed SA and a list of SA resulting from each wave (with vmax of 1 km/yr)

        :param SA_obs:
        :param SA_waves_list:
        :param vmin:
        :param vmax:
        :param Namps:
        :return:
        '''
        amp = np.linspace(amp_min, amp_max, Namps)
        args = tuple([amp] * len(SA_waves_list))
        out = itertools.product(*args)
        misfit = np.empty(Namps ** len(SA_waves_list))
        for i, amps in enumerate(out):
            SA_waves = np.zeros_like(SA_waves_list[0])
            for sa, a in zip(SA_waves_list, amps):
                SA_waves += sa * a
            misfit[i] = self.rms_region_allT(SA_obs - SA_waves, weights=weights)
        max_misfit = np.max(np.abs(misfit))
        return (max_misfit - np.reshape(misfit, [Namps] * len(SA_waves_list)))/max_misfit

    def find_best_amplitudes_from_swept(self, amp_swept, amp_min=0.1, amp_max=5, Namps=20, return_inds=False):
        ''' finds the best-fit set of amplitudes given an array of fit values

        :param amp_fits:
        :param vmin:
        :param vmax:
        :param Namps:
        :param return_inds:
        :return:
        '''
        ind_max = np.unravel_index(np.argmax(amp_swept), amp_swept.shape)
        amp = np.linspace(amp_min, amp_max, Namps)
        v_fits = []
        for i in ind_max:
            v_fits.append(amp[i])
        if return_inds:
            return v_fits, ind_max
        else:
            return v_fits

    def get_peak_phase_period_slice(self, phases, periods, corr, return_peak_location=False):
        z = np.array(corr.T)
        z = np.concatenate((z,-z), axis=1)
        phase_plt = np.linspace(0,360,len(phases)*2, endpoint=False)
        zpeak = np.max(z)
        peri,phsi = np.where(z==zpeak)
        phase_val = phase_plt[phsi[0]]
        per_val = periods[peri[0]]
        # print("Peak Correlation phase={0:.1f} degrees, period={1:.2f} yrs".format(phase_val, per_val))
        m_slice = z[:,phsi[0]]
        if return_peak_location:
            return m_slice, phase_val, per_val
        else:
            return m_slice

    def get_peak_phase_period_eachperiod(self, phases, periods, corr):
        z = np.array(corr.T)
        z = np.concatenate((z, -z), axis=1)
        return np.max(z,axis=1)

class SVNoiseModel(flows):
    def __init__(self):
        self.mana = analysis()

    def fit_lm_sd(self, lm_data, deg=1, return_real_sd=False):
        ''' fit the mean and standard deviation of a fft coefficient of a set of spherical harmonic coefficients up to l_max

        :param lm_data:
        :param deg:
        :param return_real_sd:
        :return:
        '''
        l_max = lm_data.shape[0] - 1
        l_weights = np.linspace(1, l_max + 1, l_max + 1)
        l_values = np.linspace(0, l_max, l_max + 1)
        mean_by_l = np.empty(l_max + 1)
        sd_by_l = np.empty(l_max + 1)
        for l in range(l_max + 1):
            mean_by_l[l] = np.mean(lm_data[l, :l + 1])
            sd_by_l[l] = np.std(lm_data[l, :l + 1])
        pf = np.polyfit(l_values, mean_by_l, deg, w=l_weights)
        fit = np.polyval(pf, l_values)
        pf_sd = np.polyfit(l_values, sd_by_l, deg=1)
        sd_fit = np.polyval(pf_sd, l_values)
        sd_fit[sd_fit<0.] =0.
        if return_real_sd:
            return fit, sd_fit, sd_by_l
        else:
            return fit, sd_fit

    def fit_lm_sd_in_log(self, lm_data, deg=1):
        ''' fit mean, high 1 std, low 1 std, in log space

        :param lm_data:
        :param deg:
        :return:
        '''
        mean, sd = self.fit_lm_sd(np.log10(lm_data), deg=deg)
        return 10 ** mean, 10 ** (mean - sd), 10 ** (mean + sd)

    def fit_lm_sd_in_linear(self, lm_data, deg=1):
        ''' fit mean, high 1 std, low 1 std, in linear space

        :param lm_data:
        :param deg:
        :return:
        '''
        mean, sd = self.fit_lm_sd(lm_data, deg=deg)
        return mean, mean - sd, mean + sd

    def fit_all_lm_sd(self, lm_fft, Nfft=None, deg_fits=[1, 2, 2, 2, 2], log=False):
        ''' fits mean, sdl, sdh for each Fourier coefficient

        :param Nfft:
        :param deg_fits:
        :param log:
        :return:
        '''
        if Nfft is None:
            Nfft = min(lm_fft.shape[-1], len(deg_fits))
        fits = np.empty((Nfft, lm_fft.shape[0]))
        sdls = np.empty((Nfft, lm_fft.shape[0]))
        sdhs = np.empty((Nfft, lm_fft.shape[0]))
        if log:
            for i in range(Nfft):
                fits[i, :], sdls[i, :], sdhs[i, :] = self.fit_lm_sd_in_log(np.abs(lm_fft[:, :, i]), deg=deg_fits[i])
        else:
            for i in range(Nfft):
                fits[i, :], sdls[i, :], sdhs[i, :] = self.fit_lm_sd_in_linear(np.abs(lm_fft[:, :, i]), deg=deg_fits[i])
        return fits, sdls, sdhs

    def generate_rand_lm_mags(self, mean, sd, l_max=None, m0_modifier=0.9):
        ''' generates random values for the magnitudes of a particular fft coefficient of a set of spherical harmonics

        Normal Distribution

        :param mean:
        :param sd:
        :param l_max:
        :param m0_modifier:
        :return:
        '''
        if l_max is None:
            l_max = mean.shape[0] - 1
        rand_vals = np.zeros((l_max + 1, l_max + 1))
        l = 0
        rand_vals[l:l_max + 1, l] = np.abs(np.random.normal(loc=mean[l:l_max + 1], scale=sd[l:l_max + 1])) * m0_modifier
        for l in range(1, l_max + 1):
            rand_vals[l:l_max + 1, l] = np.abs(np.random.normal(loc=mean[l:l_max + 1], scale=sd[l:l_max + 1]))
        return rand_vals

    def generate_rand_lm_phases(self, l_max):
        ''' generates random values for the phases of a particular fft coefficient of a set of spherical harmonics.

        Uniform Distribution

        :param l_max:
        :return:
        '''
        rand_phases = np.zeros((l_max + 1, l_max + 1))
        for l in range(l_max + 1):
            rand_phases[l:l_max + 1, l] = np.random.uniform(low=-np.pi, high=np.pi, size=l_max + 1 - l)
        return rand_phases

    def generate_all_rand_lm_magphase(self, lm_fft, degfit_by_fft=[1, 2, 2, 2, 2], log=False):
        ''' generates random magnitudes and phases for each Fourier coefficient for each spherical harmonic lm

        :param lm_fft:
        :param degfit_by_fft:
        :param log:
        :return:
        '''
        l_max = lm_fft.shape[0] - 1
        Nfft = lm_fft.shape[-1]
        if log:
            lm_mag = np.log10(np.abs(lm_fft))
            m0_modifier = 0.9
        else:
            lm_mag = np.abs(lm_fft)
            m0_modifier = 0.5
        rand_mags = np.zeros((l_max + 1, l_max + 1, Nfft))
        rand_phases = np.zeros((l_max + 1, l_max + 1, Nfft))
        for n in range(Nfft):
            lm_data = lm_mag[:, :, n]
            mean, sd = self.fit_lm_sd(lm_data, deg=degfit_by_fft[n])
            if not log:
                mean[mean<0.] = 0.
            rand_mags[:, :, n] = self.generate_rand_lm_mags(mean, sd, m0_modifier=m0_modifier)
            rand_phases[:, :, n] = self.generate_rand_lm_phases(l_max)
        if log:
            rand_mags = 10 ** rand_mags
        return rand_mags, rand_phases

    def generate_rand_SV(self, T, lm_fft, degfit_by_fft=[1, 2, 2, 2, 2], log=False, norm='4pi', Nth=None, normalize_to_rms=None, norm_weights=1., return_norm_ratio=False):
        ''' generates a new realization of the SV resdiual spherical harmonics across time

        :param T:
        :param lm_fft:
        :param degfit_by_fft:
        :param log:
        :param norm:
        :return:
        '''
        if Nth is None:
            Nth = lm_fft.shape[0]*2
        rand_mags, rand_phases = self.generate_all_rand_lm_magphase(lm_fft, degfit_by_fft=degfit_by_fft, log=log)
        rand_fft = rand_mags * np.exp(1j * rand_phases)
        SV_rand_sh = self.get_lm_ifft(T, rand_fft, norm=norm)
        if normalize_to_rms is not None:
            SV_rand, SV_rand_sh, norm_ratio = self.normalize_SV(SVsh_to_normalize=SV_rand_sh, Nth=Nth, rms_norm=normalize_to_rms, weights=norm_weights, return_norm_ratio=True)
        else:
            SV_rand = self.vSH2v_allT(SV_rand_sh, Nth=Nth)
        if return_norm_ratio:
            return SV_rand, SV_rand_sh, norm_ratio
        else:
            return SV_rand, SV_rand_sh

    def generate_rand_SA(self, T, lm_fft, degfit_by_fft=[1, 2, 2, 2, 2], log=False, norm='4pi', Nth=None, normalize_to_rms=None, norm_weights=1., return_norm_ratio=False):
        ''' generates a new realization of the SV resdiual spherical harmonics across time

        :param T:
        :param lm_fft:
        :param degfit_by_fft:
        :param log:
        :param norm:
        :return:
        '''
        if Nth is None:
            Nth = lm_fft.shape[0]*2
        rand_mags, rand_phases = self.generate_all_rand_lm_magphase(lm_fft, degfit_by_fft=degfit_by_fft, log=log)
        rand_fft = 1j*np.fft.fftfreq(len(rand_mags), d=T[-1]-T[0])*rand_mags * np.exp(1j * rand_phases)
        SV_rand_sh = self.get_lm_ifft(T, rand_fft, norm=norm)
        if normalize_to_rms is not None:
            SV_rand, SV_rand_sh, norm_ratio = self.normalize_SV(SVsh_to_normalize=SV_rand_sh, Nth=Nth, rms_norm=normalize_to_rms, weights=norm_weights, return_norm_ratio=True)
        else:
            SV_rand = self.vSH2v_allT(SV_rand_sh, Nth=Nth)
        if return_norm_ratio:
            return SV_rand, SV_rand_sh, norm_ratio
        else:
            return SV_rand, SV_rand_sh

    def get_lm_ifft(self, T, lm_fft, norm='4pi'):
        ''' computes the inverse Fourier transform across time for a set of spherical harmonics

        :param T:
        :param lm_fft:
        :param norm:
        :return:
        '''
        l_max = lm_fft.shape[0] - 1
        lm_sh = np.zeros((len(T), 2, l_max + 1, l_max + 1))
        for l in range(l_max + 1):
            for m in range(l + 1):
                f = np.zeros((len(T)), dtype='complex')
                Nfreq = (lm_fft.shape[2] - 1) // 2
                f[:Nfreq + 1] = lm_fft[l, m, :Nfreq + 1]
                f[-Nfreq:] = lm_fft[l, m, -Nfreq:]
                ifft = np.fft.ifft(f, n=len(T))
                lm_sh[:, 0, l, m] = ifft.real
                lm_sh[:, 1, l, m] = ifft.imag
        return lm_sh

    def get_lm_fft(self, T, shcoeffs_t, Nfft=5, l_max=14, norm='4pi', return_l_values=False):
        ''' computes the Fourier transform across time for a set of spherical harmonics

        :param T:
        :param shcoeffs_t:
        :param Nfft:
        :param l_max:
        :param norm:
        :param return_l_values:
        :return:
        '''
        l_arr = min(shcoeffs_t.shape[2], l_max + 1)
        lm_fft = np.zeros((l_max + 1, l_max + 1, Nfft), dtype='complex')
        Nfreq = (Nfft - 1) // 2
        for l in range(l_arr):
            for m in range(l + 1):
                fft = np.fft.fft(shcoeffs_t[:, 0, l, m] + shcoeffs_t[:, 1, l, m] * 1j)
                ni = (len(fft) - 1) // 2 + 1
                lm_fft[l, m, 0] = fft[0]
                lm_fft[l, m, 1:Nfreq + 1] = fft[1:Nfreq + 1]
                lm_fft[l, m, -Nfreq:] = fft[-Nfreq:]
        return lm_fft

    def unroll_phase(self, phase):
        ''' takes a list of phases across time in the range (-pi, pi) and unrolls it into a continuous function of unlimited range

        :param phase:
        :return:
        '''
        p = phase
        for i in range(len(p) - 1):
            dp = p[i + 1] - p[i]
            if dp > np.pi:
                phase[i + 1:] += -2 * np.pi
            elif dp < -np.pi:
                p[i + 1:] += 2 * np.pi
        return p

    def crop_pwn(self, m, freq, pwn, m_max, T_min, T_max, return_indexes=False):
        ''' crops a period-wavenumber transformation into only the desired range for smaller storage.

        :param pwn:
        :param m_max:
        :param T_min:
        :param T_max:
        :return:
        '''
        im_max = np.where(m > m_max)[0][0]
        im_min = np.where(m > -m_max)[0][0]
        it_min = np.where(freq > 1 / T_max)[0][0]
        it_max = np.where(freq > 1 / T_min)[0][0]
        pwn_ind = ((it_min, it_max+(it_max-it_min)), (im_min, im_max))
        m_out = m[im_min:im_max]
        freq_out = freq[it_min:it_max]
        pwn_out = pwn[pwn_ind[0][0]:pwn_ind[0][1], pwn_ind[1][0]:pwn_ind[1][1]]
        if return_indexes:
            return m_out, freq_out, pwn_out, (im_min, im_max), (it_min, it_max), pwn_ind
        else:
            return m_out, freq_out

    def get_magphase(self, data_in):
        '''

        :param data_in:
        :return:
        '''
        mag = np.abs(data_in)
        if isinstance(data_in, (list, np.ndarray)):
            phase = self.unroll_phase(np.arctan2(data_in.imag,data_in.real))
        else:
            phase = np.arctan2(data_in.imag,data_in.real)
        return mag, phase

    def get_lm_magphase(self, lm_fft):
        '''

        :param lm_fft:
        :return:
        '''
        mag = np.zeros_like(lm_fft, dtype=float)
        phase = np.zeros_like(lm_fft, dtype=float)
        for l in range(lm_fft.shape[0]):
            for m in range(l+1):
                for i in range(lm_fft.shape[-1]):
                    mag[l,m,i], phase[l,m,i] = self.get_magphase(lm_fft[l,m,i])
        return mag, phase

    def normalize_SV(self, SVsh_to_normalize=None, SV_to_normalize=None, rms_norm=None, SV_real=None, weights=1., Nth=None, l_max=None, return_norm_ratio=False):
        ''' normalizes the rms power of one dataset to match another

        :param SV_real:
        :param SV_to_normalize:
        :param weights:
        :param SVsh_to_normalize:
        :return:
        '''
        if rms_norm is None:
            if SV_real is None:
                raise ValueError('Must specify either rms_norm or a SV dataset to normalize to')
            else:
                rms_norm = self.mana.rms_region_allT(SV_real, weights=weights)
        if SVsh_to_normalize is None:
            if SV_to_normalize is None:
                raise ValueError('must specify either SV_to_normalize or SVsh_to_normalize or both')
            else:
                SVsh_to_normalize = self.v2vSH_allT(SV_to_normalize, Nth=Nth, l_max=l_max)
        if SV_to_normalize is None:
            if SVsh_to_normalize is None:
                raise ValueError('must specify either SV_to_normalize or SVsh_to_normalize or both')
            else:
                SV_to_normalize = self.vSH2v_allT(SVsh_to_normalize, Nth=Nth, l_max=l_max)
        SV_2norm_rms = self.mana.rms_region_allT(SV_to_normalize, weights=weights)
        norm_ratio = rms_norm/SV_2norm_rms
        if return_norm_ratio:
            return SV_to_normalize*norm_ratio, SVsh_to_normalize*norm_ratio, norm_ratio
        else:
            return SV_to_normalize*norm_ratio, SVsh_to_normalize*norm_ratio

    def compute_many_SVsr_pwn(self, T, SVr, N, pwn_weights=None, Nth=None, Nfft=5, degfit=[1,2,2,2,2], logfit=False, l_max=14, m_max=14, T_min=2.5, T_max=24, norm_weights=1.):
        if pwn_weights is None:
            th, ph = self.get_thvec_phvec_DH(Nth=SVr.shape[1], l_max=l_max)
            lat = th-90
            sigmath = 16
            pwn_weights = ffit.hermite_fun(lat/sigmath, 0)
        if Nth is None:
            Nth = SVr.shape[1]
        SVr_rms = self.mana.rms_region_allT(SVr, weights=norm_weights)
        SVrsh = self.v2vSH_allT(SVr)
        SVr_fft = self.get_lm_fft(T, SVrsh, Nfft=Nfft, l_max=l_max)
        SVsr, _ = self.generate_rand_SV(T, SVr_fft, degfit_by_fft=degfit, log=logfit, Nth=Nth, normalize_to_rms=SVr_rms, norm_weights=norm_weights, return_norm_ratio=False)
        SVsr_eq = self.mana.weighted_mean_region_allT(SVsr, th=th, weights=pwn_weights)
        m, freq, SVsr_pwn  = self.mana.compute_frequency_wavenumber(SVsr_eq, T)
        m_save, freq_save, pwn, m_ind, freq_ind, pwn_ind = self.crop_pwn(m,freq, SVsr_pwn, m_max, T_min, T_max, return_indexes=True)
        Nm = len(m_save)
        Nt = len(freq_save)
        pwn_all = np.empty((N, Nt*2, Nm), dtype=np.float)
        N10 = max(N//10,1)
        for i in range(N):
            if i%N10 == 0:
                print('on step {}/{}'.format(i,N))
            SVsr, _ = self.generate_rand_SV(T, SVr_fft, degfit_by_fft=degfit, log=logfit, Nth=Nth, normalize_to_rms=SVr_rms, norm_weights=norm_weights, return_norm_ratio=False)
            SVsr_eq = self.mana.weighted_mean_region_allT(SVsr, th=th, weights=pwn_weights)
            _, _, SVsr_pwn  = self.mana.compute_frequency_wavenumber(SVsr_eq, T)
            pwn_all[i,:,:] = np.abs(SVsr_pwn[pwn_ind[0][0]:pwn_ind[0][1], pwn_ind[1][0]:pwn_ind[1][1],])
        return m_save, freq_save, pwn_all

    def save_computed_noise(self, m,freq,pwn_all, filename='computed_noise.m'):
        dill.dump((m,freq,pwn_all),open(filename,'wb'))

    def load_computed_noise(self, filename='computed_noise.m'):
        m,freq,pwn_all = dill.load(open(filename,'rb'))
        return m,freq,pwn_all

    def compute_pwn_percentile(self, pwn_all, p):
        return np.percentile(pwn_all, p, axis=0)

