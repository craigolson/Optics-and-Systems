########################################################################
########################################################################
"""
This class supports calculation of line-of-sight motion-induced modulation
transfer function (MTF) from either time-series or power spectrum data. 
The basic object represents a time series from which a brute force histogram-
based point-spread function can be calculated for a given integration time.
The PSF is used to derive the MTF 

For details and author contact information please see the following reference:

- S. Craig Olson, David Gaudiosi, Andrew Beard, Rich Gueler, 
"Statistical evaluation of motion-based MTF for full-motion video using 
the Python-based PyBSM image quality analysis toolbox", Proc. SPIE 10650, 
Long-Range Imaging III, 106500L (11 May 2018); doi: 10.1117/12.2305406; 
https://doi.org/10.1117/12.2305406 

The class is intended to support the PyBSM imaging model outlined here:

- LeMaster, D. and Eismann, M., “pyBSM: A Python package for modeling imaging 
systems,” Proc. SPIE 10204; Long-Range Imaging II, 1020405 (2017), 
doi: 10.1117/12.2262561

The calculation methods are based entirely on openly available formalism primarily 
fromthe following two sources:

- Kopeika, N., [A System Approach to Imaging], SPIE Press, (1998).

- Youngworth, R., Gallagher, B.B., and Stamper, B.L., “An overview of power 
spectral density (PSD) calculations,” Proc. SPIE 5869; Optical Manufacturing 
and Testing VI, 58690U, (2005) doi: 10.117/12.618478


Note that in order to use this class with the existing PyBSM framework an auxiiliary
motion MTF helper function is required; this code will be posted in the near future to
reproduce the paper results in the first reference above.

An interesting follow-on exercise would be to implement the method of moments technique
outlined in Kopeika's text and paper for a potentially faster and analytically
tractable method of computing MTF

Currently the code only supports a 1-D MTF using single-dimension vectors (although X and y
components are demosntrated here). PyBSM can accomodate fully 2D MTF and PSF arrays.  
This was not implemented in this version to maintain decent computational efficiency.

MIT License

Copyright (c) 2018 Stephen Craig Olson

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""
########################################################################
########################################################################

from scipy import interpolate
import numpy as np
from matplotlib import pyplot as plt

#from scipy.integrate import simps, cumtrapz
#from scipy.signal import periodogram as pd

import logging
import inspect
import os
import pandas as pd


class lineofsight:
    """
    Class lineofsight represents a time series of line-of-sight motion data
    in units of image motion at the focal plane. 
    """
    def __init__(self, amplitudedata=[], timestep=0.001, 
                 frequencies=[], PSD=[], numberofpoints = 10000,
                 sourcetype='', kinematictype=''):
        """
        Defines a default time series line of sight motion dataset
        randomized to 20 um Gaussian time series (white noise).
        A user-input amplitude array of motion of any number of points can be input as argument 
        If amplitudedata has is missing or is an empty list, a 20um / 10000pt random array is created
        First argument of amplitudedata is the sigma value for Gaussian motion
        Second argument of amplitudedata is the number of total points.
        The timestep argument is in seconds
        """
        self.g      = 9.80665 # Standard value of gravitational acceleration [m/s^2]
        
        if (sourcetype.lower() == 'psd') or (sourcetype.lower() == 'sd'):
            if (not frequencies) or (not PSD):
                self.displayWarning('sourcetype ''psd'' or ''sd'' requires two non-empty lists or arrays.')
                self.numpts         = []
                self.timestep       = []
                self.times          = []
                self.maxtime        = []
                self.freqstep       = []
                self.frequencies    = []
                self.maxfreq        = []
                self.amplitudes     = []
            elif (sourcetype.lower() == 'sd'):
                self.MakeFromSpectralDensity(frequencies,PSD,numberofpoints)
            else:
                self.MakeFromAccelPSD(frequencies,PSD,numberofpoints)
            
        elif (not sourcetype) or (sourcetype.lower() == 'random') or (sourcetype.lower() == 'amplitude'):
            # warning messages
            ampwarningmsg = ''.join("WARNING: 'amplitude' source type is not compatible with amplitudedata provided.  "
                          "Generating random data with provided ampliude data.")

            rndwarningmsg = ''.join("WARNING: 'random' source type is not compatible with amplitudedata provided.  "
                  "Using amplitude data directly.")
            
            kinwarningmsg = ''.join("WARNING: Incorrect kinematic type.")
            
            # amplitude array
            if len(amplitudedata) == 0:
                self.amplitudes = self.MakeRandomLOS()
                if sourcetype.lower() == 'amplitude':
                    self.displayWarning(ampwarningmsg)                    
            elif len(amplitudedata) == 1:
                self.amplitudes = self.MakeRandomLOS(amplitudedata,numberofpoints)
                if sourcetype.lower() == 'amplitude':
                    self.displayWarning(ampwarningmsg)                    
            elif len(amplitudedata) == 2:
                self.amplitudes = self.MakeRandomLOS(amplitudedata[0], amplitudedata[1])
                if sourcetype.lower() == 'amplitude':
                    self.displayWarning(ampwarningmsg)                    
            else:
                N   = len(amplitudedata)
                t   = timestep * np.arange(0,N)
                # use the argument array as as the time series
                if (kinematictype.lower() == 'displacement') or (kinematictype.lower() == ''):
                    self.amplitudes = amplitudedata
                elif kinematictype.lower() == 'velocity':
                    dispdata        = cumtrapz(amplitudedata,t)
                    dispdata        = np.insert(dispdata,0,0)
                    dispdata        -= np.mean(dispdata)
                    self.amplitudes = dispdata
                elif kinematictype.lower() == 'acceleration':
                    veldata         = cumtrapz(amplitudedata,t)
                    veldata         = np.insert(veldata,0,0)
                    veldata         -= np.mean(veldata)
                    dispdata        = cumtrapz(veldata,t)
                    dispdata        = np.insert(dispdata,0,0)
                    dispdata        -= np.mean(dispdata)
                    self.amplitudes = dispdata
                else:
                    self.amplitudes = []
                    self.displayWarning(kinwarningmsg)                    
                    
                if sourcetype.lower() == 'random':
                    self.displayWarning(rndwarningmsg)                    

            # time arrays
            self.numpts         = len(self.amplitudes)  
            self.timestep       = timestep
            self.times          = timestep * np.arange(0,self.numpts)
            self.maxtime        = self.numpts * self.timestep
            
            # frequency arrays
            self.maxfreq        = 1 / self.timestep;
            self.freqstep       = self.maxfreq / (self.numpts-1);
            #self.frequencies = (self.freqstep * np.arange(0, self.numpts))[0:self.numpts/2]
            self.frequencies    = np.fft.fftfreq(self.numpts, self.timestep)[0:self.numpts/2]

        else:
            self.displayWarning("source type must be 'random','amplitude', or 'psd'")
            self.numpts         = []
            self.timestep       = []
            self.times          = []
            self.maxtime        = []
            self.freqstep       = []
            self.frequencies    = []
            self.maxfreq        = []
            self.amplitudes     = []
            
        
    def GetFrameInfo(self, tint=0.001):
        """Returns the number of frames in the los sequence for the given integration time
        tint in seconds
        """
        numvals = int(np.round(tint / self.timestep))
        numframes = self.numpts/numvals
        # want to extend this method to return array of starting indices for each frame number
        return numframes
    
    def MakeRandomLOS(self, amplitudesigma=0.000020, numberofpoints=10000):
        """
        Makes a Gaussian random time series defaulting to 20E-6 m (20 um) sigma w/10000 points
        """
        # this could be more integrated with the class, so that it can default
        # to random when instantiated
        randoms = np.random.normal(0, amplitudesigma, numberofpoints )
        return randoms
    
    def MakeFromAccelPSD(self, input_freq, input_PSD, numberofpoints=10000):
        """
        Creates a displaement time series [units of m] from a given acceleration power spectral density input
        Two input arrays:
            frequency array     = input frequencies [Hz]
            acceleration PSD    = acceleration power spectral density [g^2 / Hz]

        Optional input:
            number of points    = number of points in displacement time series
        """
        if (not input_freq) or (not input_PSD):
            print('MakeFromPSD requires two non-empty lists or arrays.')
            return
        
        N       = numberofpoints
        
        # Create arrays from inputs
        f_in    = np.array( input_freq, dtype=np.double )
        PSD_in  = np.array( input_PSD, dtype=np.double )
        
        # Create time and frequency arrays
        # Set sampling rate sufficient for accurate RMS calculations
        # Dependent on number of points and max input frequency
        dt      = 1 / ( 5 * np.log10( N ) * f_in.max() )
        f_2s    = np.fft.fftfreq( int( N ),dt )
        df      = 1/( N * dt )
        f       = f_2s[ 0:int( N / 2 ) ]
        f[0]    = np.power( np.finfo( f.dtype ).eps, 0.1 )  # Set f[0] such that all future calculations avoid divide-by-zero error
    
        t       = np.arange( 0, N ) * dt
    
        # interpolate input PSD 
        logPSD_fun = interpolate.interp1d( np.log10(f_in), 
                                       np.log10(PSD_in),
                                       kind = 'linear', 
                                       bounds_error = False,
                                       fill_value = np.finfo( f.dtype ).min )
        # Create PSD interpolated array
        PSD     = np.power(10,logPSD_fun(np.log10(f)))
        # Create 2-sided PSD array for ifft
        PSD_2s  = np.concatenate((PSD,[0],PSD[-1:0:-1])) / 2
        
        # Calculate velocity spectral density
        VSD     = np.divide( PSD, np.power( 2. * np.pi * f, 2 ) )
        # Create 2-sided PSD array for ifft
        VSD_2s  = np.concatenate((VSD,[0],VSD[-1:0:-1])) / 2
        
        # Calculate displacement spectral density
        DSD     = np.divide( PSD, np.power( 2. * np.pi * f, 4 ) )
        # Create 2-sided PSD array for ifft
        DSD_2s  = np.concatenate((DSD,[0],DSD[-1:0:-1])) / 2
        
        # Create arrays from random time series 
        # (normal distribution, mean = 0, variance = 1)
        # and magnitude of PSD
        randt   = np.random.randn(PSD_2s.size)
        
        # Calculate acceleration, velocity, and displacement time series
        # N.B Time series' CANNOT be calculated from one another 
        # (i.e. accel does NOT equal d(velocity)/dt)
        # They are statistically equivalent 
        # (i.e. rms value of accel = rms value of d(velocity)/dt)
        # accel and vel reserved for future use
        
        randfft = np.fft.fft( randt )
        accel   = self.g * np.real( np.fft.ifft( np.multiply( np.exp( -1j * np.angle( randfft ) ), PSD_2s.size * np.sqrt( PSD_2s * df ) ) ) )

        vel     = self.g * np.real( np.fft.ifft( np.multiply( np.exp( -1j * np.angle( randfft ) ), PSD_2s.size * np.sqrt( VSD_2s * df ) ) ) )

        disp    = self.g * np.real( np.fft.ifft( np.multiply( np.exp( -1j * np.angle( randfft ) ), PSD_2s.size * np.sqrt( DSD_2s * df ) ) ) )

        # save arrays as class properties
        self.numpts         = int( N )
        self.timestep       = dt
        self.times          = t
        self.maxtime        = self.times.max()
        self.freqstep       = df
        self.frequencies    = f
        self.maxfreq        = f.max()
        self.amplitudes     = disp
        self.__disp         = disp
        self.__vel          = vel
        self.__accel        = accel
        self.__DSD          = DSD
        self.__VSD          = VSD
        self.__PSD          = PSD
        self.__f_2s         = f_2s
#        self.frequencies    = np.fft.fftfreq(self.numpts, self.timestep)[0:self.numpts/2]
        
        return

    def MakeFromSpectralDensity(self, input_freq, input_SD, numberofpoints=100000):
        """
        Creates a displaement time series [units dependent on spectral density units] 
        from a given spectral density input
        Two input arrays:
            frequency array     = input frequencies [Hz]
            Spectral Density    = spectral density [<units>^2 / Hz]

        Optional input:
            number of points    = number of points in displacement time series
        """
        if (not input_freq) or (not input_SD):
            print('MakeFromSpectralDensity requires two non-empty lists or arrays.')
            return
        
        N       = numberofpoints
        
        # Create arrays from inputs
        f_in    = np.array( input_freq, dtype=np.double )
        SD_in   = np.array( input_SD, dtype=np.double )
        
        # Create time and frequency arrays
        # Set sampling rate sufficient for accurate RMS calculations
        # Dependent on number of points and max input frequency
        dt      = 1 / ( 5 * np.log10( N ) * f_in.max() )
        f_2s    = np.fft.fftfreq( int( N ),dt )
        df      = 1/( N * dt )
        f       = f_2s[ 0:int( N / 2 ) ]
        f[0]    = np.power( np.finfo( f.dtype ).eps, 0.1 )  # Set f[0] such that all future calculations avoid divide-by-zero error
    
        t       = np.arange( 0, N ) * dt
    
        # interpolate input SD 
        logSD_fun = interpolate.interp1d( np.log10(f_in), 
                                       np.log10(SD_in),
                                       kind = 'linear', 
                                       bounds_error = False,
                                       fill_value = np.finfo( f.dtype ).min )
        # Create SD interpolated array
        SD     = np.power(10,logSD_fun(np.log10(f)))
        # Create 2-sided SD array for ifft
        SD_2s  = np.concatenate((SD,[0],SD[-1:0:-1])) / 2
        
        # Create arrays from random time series 
        # (normal distribution, mean = 0, variance = 1)
        # and magnitude of SD
        randt   = np.random.randn(SD_2s.size)
        # Ensure random series has zero mean
        randt   = randt - np.mean(randt)
        
        # Calculate acceleration, velocity, and displacement time series
        # N.B Time series' CANNOT be calculated from one another 
        # (i.e. accel does NOT equal d(velocity)/dt)
        # They are statistically equivalent 
        # (i.e. rms value of accel = rms value of d(velocity)/dt)
        # accel and vel reserved for future use
        
        randfft = np.fft.fft( randt )
        amp     = np.real( np.fft.ifft( np.multiply( np.exp( -1j * np.angle( randfft ) ), SD_2s.size * np.sqrt( SD_2s * df ) ) ) )

        # save arrays as class properties
        self.numpts         = int( N )
        self.timestep       = dt
        self.times          = t
        self.maxtime        = self.times.max()
        self.freqstep       = df
        self.frequencies    = f
        self.maxfreq        = f.max()
        self.amplitudes     = amp
        
        return

    def getDebugInfo(self):
        debugTuple = (self.__disp,self.__vel,self.__accel,self.__DSD,self.__VSD,self.__PSD,self.__f_2s)
        return debugTuple
    
    def PowerSpectrum(self, PStype='acceleration'):
        """
        Returns the 1-sided FFT-based power spectrum (PS) of the amplitude data
        in squared SI units . Returns array of length amplitude.
        """
        
        dispPowerSpectrum   = np.power( np.absolute( np.fft.fft( self.amplitudes ) ) / self.amplitudes.size , 2 )
        # Zero values under 1e-10
        dispPowerSpectrum[np.nonzero(dispPowerSpectrum < 1e-18)] = 0
        # Convert to one sided PS
        dispPowerSpectrum   = 2 * dispPowerSpectrum[ 0:int( self.numpts / 2 ) ]
        velPowerSpectrum    = np.multiply( dispPowerSpectrum, np.power( 2. * np.pi * self.frequencies, 2 ) )
        accelPowerSpectrum  = np.multiply( velPowerSpectrum, np.power( 2. * np.pi * self.frequencies, 2 ) )
        
        if PStype.lower() == 'displacement':
            PS = dispPowerSpectrum
        elif PStype.lower() == 'velocity':
            PS = velPowerSpectrum
        elif PStype.lower() == 'acceleration':
            PS = accelPowerSpectrum
        else:
            self.displayWarning('Invalid power spectrum type.  Returning empty array.')
            PS = []
            
        return PS
    
    def PowerSpectralDensity(self, PSDtype='acceleration'):
        """
        Returns the FFT-based power spectral density (PSD) of the amplitude data
        in units of amplitude^2 / Hz. Returns array of length amplitude / 2
        """
        if PSDtype.lower() == 'displacement':
            PS = self.PowerSpectrum(PStype = 'displacement')
        elif PSDtype.lower() == 'velocity':
            PS = self.PowerSpectrum(PStype = 'velocity')
        elif PSDtype.lower() == 'acceleration':
            PS = self.PowerSpectrum(PStype = 'acceleration')
        else:
            self.displayWarning('Invalid power spectral density type.  Returning empty array.')
            PS = []
            
        return PS / self.freqstep

    def BandLimitedRMS(self, tint=0.001):
        """
        Returns the integrated root-mean-square (sigma) motion by integrating
        under the PSD curve from infinity frequency (zero time) down to the specified
        frequency (integration time). Useful for comparing the resulting RMS value
        to the Gaussian equivalent jitter MTF and distributions."
        """
        # 
        # report frequency range of PSD integral based on integration time
        # and maximum frequency in time series / PSD
        #
        # will need to calculate index numbers of PSD based on some sort of 
        # calculation here
        # round (timestep * index number) = desired tint / min frequency
        # The min frequency index will be the desired tint / timestep rounded down
        # to nearest int
        
        # do some error checking here to make sure tint is not smaller than max freq
        
        if tint == 0: 
            t_index = 0;
        else:
            t_index = int(1 / (tint * self.freqstep));
        psd = self.PowerSpectralDensity('displacement');
        rms1 = np.sqrt(  np.trapz(psd[t_index:], dx=self.freqstep)  );
        #rms2 = np.sqrt( self.freqstep * np.sum(psd[t_index:])   );
        
        return rms1


    def PointSpreadFunction(self, startpoint=0, tint=0.005, interpolationpoints=1000, histogrambins=200):
        """
        numInterp = 1000; # number of splien interpolation points for PSF
        numbins = 250; # number of bins to downsample PSF into for histogram
        """
        # define starting point in the full time series data
        st = max(0, startpoint)        
        
        # count the number of points needed for integration time and specific frame number
        pointsperframe = int(np.ceil(tint / self.timestep));
        if st + pointsperframe > self.numpts: st = self.numpts - pointsperframe - 1
                
        # extract the as-sampled frame motion dataset (small number of points)
        # take out the DC bias for each frame
        framedataLowRaw = self.amplitudes[st:st + pointsperframe]
        framedataLow = framedataLowRaw - np.mean(framedataLowRaw)
        timedataLow = np.arange(0, tint, self.timestep)
        
        # create an oversampled frame motion dataset (interpolationpoints)
        if len(framedataLowRaw) < interpolationpoints:
            timedataHigh = np.linspace(0, tint, num=interpolationpoints)
            framedataHigh = np.interp(timedataHigh, timedataLow, framedataLow)
        
            #Cubic spline interpolation of low-sampled data
            framesplineHigh = interpolate.CubicSpline(timedataLow, framedataLow, extrapolate=False)
            frameCurveHigh = framesplineHigh(timedataHigh)
        else:
            timedataHigh = timedataLow
            framedataHigh = framedataLow
            frameCurveHigh = framedataLow
             
        # Define ranges and zero-padding span for PSF (needed downstream for FFT sampling)
        mxMotion = np.max(framedataHigh)
        mnMotion = np.min(framedataHigh)
        mxMtnLim = np.max(np.abs(np.concatenate(([mnMotion],[mxMotion]))))
                      
        span = 5;
        histY, binsX = np.histogram(frameCurveHigh, bins=histogrambins, range=(-span*mxMtnLim, span*mxMtnLim) ); # assumes mnMotion is negative
        frameSpatialX = binsX[0:len(binsX)-1];
        framePSF = histY / (1.0 * max(histY));  #typecast to real
        
        # make this function return two arrays: spatial array and PSF array
        return pointsperframe, timedataLow, framedataLow, timedataHigh, framedataHigh, frameCurveHigh, framePSF, frameSpatialX

    def MotionMTF(self, psfArray=[], psfSpatial=[], startpoint=0, tint=0.005):
        """
        docstring
        """
        if (len(psfArray)==0 or len(psfSpatial)==0) and (len(self.amplitudes)==0):
            self.displayWarning('MotionMTF() requires the two PSF arguments or populated amplitude data.')
        elif (len(psfArray)==0 or len(psfSpatial)==0):
            ppf, tdLow, fdLow, tfHigh, fdHigh, fCHigh, fPSF, fX = self.PointSpreadFunction(startpoint,tint)
            psfArray = fPSF
            psfSpatial = fX
            
        otfRaw = np.fft.fft(psfArray);
        psfstep = psfSpatial[1]-psfSpatial[0];
        mtfRaw = np.absolute(otfRaw);
        mtf= mtfRaw / mtfRaw[0];
        spatialfreqstep = 1/( psfArray.size * psfstep )
        spatialFreqs = np.arange(0,spatialfreqstep*psfArray.size, spatialfreqstep)
        return spatialFreqs[0:(len(spatialFreqs)/2+1)], mtf[0:(len(mtf)/2+1)]
    
    def SaveMotionMTF(self, psfArray, psfSpatial, startpoint=0, tint=0.005, 
                      efl = 1, filename = 'temp.csv',spatial='mrad'):
        """
        docstring
        """
        header1         = 'Spatial Frequency (cy/' + spatial + ')'
        header2         = 'Single Frame 1D MTF'
        hdr             = header1 + ', ' + header2
        mtffreqs, mtf   = self.MotionMTF(psfArray, psfSpatial, startpoint, tint)
        mtf             = mtf / efl
        np.savetxt(filename,zip(mtffreqs,mtf),fmt='%.6g',delimiter=',',header=hdr)
        return 0
    
    def displayWarning( self, message ):
        filename,line_number,function_name,lines,index = inspect.getframeinfo(inspect.currentframe().f_back.f_back)
        logging.warning(" %s in %s (line %i): %s \n%s\n" % (
                function_name, 
                os.path.basename(filename), 
                line_number,
                lines[0].strip(),
                message                    
                ))                
                
######################################################################
######################################################################

if __name__ == '__main__':

    # import example data series from CSV file
    # 3-columns: time, XLOS, YLOS
    # LOS data are in mm of displacement at the focal plane
    read_data=pd.read_csv(r'C:\Users\craig\Perforce\craig_LAP-8580GH2_1653\depot\OpticsAndSensors\Tools\python\Data\XY_LOS_TimeSeries.csv')
    dataAll=(read_data.values).transpose()
    tdat=(dataAll.transpose())[:,0]
    xdat=(dataAll.transpose())[:,1]
    ydat=(dataAll.transpose())[:,2]

    # calculate time step from data; should be 0.1 ms
    tstep = tdat[1]-tdat[0]

    # make LOS objects
    losX = lineofsight(3*xdat, tstep)
    losY = lineofsight(3*ydat, tstep)
  
    # make example LOS with pure sine wave for validation
    # amplitude 50 um (0.05mm), frequency 50 Hz
    # same time sampling as example data
    # make a Y sine with a random phase for fun
    sin1x = 0.05*np.sin(2*3.14159*50*losX.times);
    losSinX = lineofsight(sin1x, tstep)
    rphase = np.random.uniform(-3.14159, 3.14159)
    sin1y = 0.05*np.sin(2*3.14159*50*losY.times + rphase);
    losSinY = lineofsight(sin1y, tstep)
    
    # make random Gaussian power spectrum
    losGX = lineofsight(amplitudedata=[0.015], timestep=tstep, numberofpoints=100000)
    losGY = lineofsight(amplitudedata=[0.0008], timestep=tstep, numberofpoints=100000)

    # make time series from amplitude power spectrum
    # define PSD envelope same in X, Y, but create different instances for each
	# envXfreqs = [10,    100,  1000,     2000]
	# envX =      [0.00002, 0.000004, 0.000004,    0.00002]
    envXfreqs = [1,    10,  100, 59.9, 60, 60.1, 1000]
    envX =      [1e-7, 1e-5, 1e-5, 1e-5, 5e-5, 1e-5 ,1e-10]
    losPSDX = lineofsight(frequencies=envXfreqs, PSD=envX, sourcetype='SD', numberofpoints=10000)
    losPSDY = lineofsight(frequencies=envXfreqs, PSD=envX, sourcetype='SD', numberofpoints=10000)

    envXfreqs2 =    [0.1, 1, 10, 100, 1000, 2000]
    envX2 =         [1e-2, 1e-3, 1e-5, 1e-7, 1e-9, 1e-9]
    losPSD2X = lineofsight(frequencies=envXfreqs2, PSD=envX2, sourcetype='SD', numberofpoints=10000)
    losPSD2Y = lineofsight(frequencies=envXfreqs2, PSD=envX2, sourcetype='SD', numberofpoints=10000)

    #####################################################################
    # Pick which LOS object we are using
    los_tempX = losGX;
    los_tempY = losGY;
    #####################################################################
    
    # Change variable here to select which time series example is processed
    # choose default integration time
    intTime = 0.0025
    
    # choose random starting index (not frame number)
    fstart = np.random.randint(0, los_tempX.numpts)
    
    # calcualte a bunch of stuff for X and Y LOS data series
    nptsX, t1x, f1x, t2x, f2x, fcx, psf1x, psfx = los_tempX.PointSpreadFunction(fstart, intTime)
    sf1x, mtf1x = losX.MotionMTF(psf1x, psfx);
    nptsy, t1y, f1y, t2y, f2y, fcy, psf1y, psfy = los_tempY.PointSpreadFunction(fstart, intTime)
    sf1y, mtf1y = losY.MotionMTF(psf1y, psfy);

    # Generate some output stats for validation / sanity check        
    print
    print("Data index start:\t\t" + repr(fstart));
    print("Integration Time:\t\t" + repr(intTime) + " seconds");
    print("Lower-bound Freq:\t\t" + repr(1/intTime) + " Hz");
    print("Number of Frames in Data:\t" + repr(los_tempX.GetFrameInfo(intTime) ));
    print
    print("Band-limited RMS:\t\t" + repr(los_tempX.BandLimitedRMS(intTime) ) + ",\t"+repr(los_tempY.BandLimitedRMS(intTime) )+" mm"  )
    print("Raw StdDev of amplitudes:\t" + repr(np.std(los_tempX.amplitudes))+",\t"+repr(np.std(los_tempY.amplitudes)) + " mm")
    print("Full-Band PSD Integral:\t\t" + repr(los_tempX.BandLimitedRMS(0) )+",\t"+ repr(los_tempY.BandLimitedRMS(0) ) + " mm"  )
    print
    
    # now plot stuff!
    # first data bout the entire time series
    plt.figure(1,figsize=(14,4));
    plt.subplot(121);
    plt.xlabel('Time (s)');
    plt.ylabel('Time Data (amplitude in mm)')
    plt.title('Time Series LOS Data')
    plt.plot(los_tempX.times,los_tempX.amplitudes,los_tempY.times,los_tempY.amplitudes )
    plt.subplot(122);
    plt.xlabel('Frequency (Hz)');
    plt.ylabel('PSD (mm^2 / Hz)')
    plt.title('Power Spectrum (FPA Displacement)')
    plt.xlim(0.1, 2e3)
    plt.ylim(1e-12, 1e-1)
    plt.loglog(los_tempX.frequencies,los_tempX.PowerSpectrum('displacement'),los_tempY.frequencies,los_tempY.PowerSpectrum('displacement') )

    # Frame-specific data
    plt.figure(2, figsize=(14,4));

    # 1D LOS traces
    plt.subplot(121);
    plt.xlabel('Time (ms)')
    plt.ylabel('LOS Motion (mm)')
    plt.title('LOS Frame Motion')
    plt.plot(1000*t1x, f1x, 'o', 1000*t2x, fcx, '-', 1000*t1y, f1y, 'o', 1000*t2y, fcy, '-')
    # 2D LOS Plot
    plt.subplot(122)
    plt.title('2D Intraframe Line-of-Sight Trace')
    plt.xlabel('Intraframe Motion X (um)')
    plt.ylabel('Intraframe Motion Y (um)')
    # get rid of NaNs from Cubicspline extrapolation
    ftx=fcx[~np.isnan(fcx)]
    fty=fcy[~np.isnan(fcy)]
    lo=1000*min(ftx.min(),fty.min())
    hi=1000*max(ftx.max(),fty.max())
    plt.xlim(lo,hi)
    plt.ylim(lo,hi)
    #plt.plot(1000*f1x, 1000*f1y,'o')
    plt.plot(1000*ftx,1000*fty,1000*f1x, 1000*f1y,'o')
    plt.figure(3, figsize=(14,4));
    
    # PSF
    plt.subplot(121);
    plt.xlabel('Image Plane Dimension (um)')
    plt.ylabel('LOS motion (um)')
    plt.title('Point-Spread Function')
    plt.plot(1000*psfx, psf1x,1000*psfy, psf1y)

    # MTF Plot
    plt.subplot(122);
    plt.xlabel('Spatial Frequency (cy/mm)')
    plt.ylabel('Modulation Transfer Fn.')
    plt.title('Single-Frame 1D MTF')
    plt.xlim(0,200)
    plt.plot(sf1x, mtf1x,'-',sf1y, mtf1y,'-')

    plt.show()


    