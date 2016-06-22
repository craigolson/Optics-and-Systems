# mtf.py file contains generic optical and other EOIR system MTF functions
# for general purpose use.
#
# file initiated C. Olson
# 12/13/2013

"""

Provides basic functions for computing optical modulation transfer function (MTF)
for circular and annular (obscured) aberrated pupils. 

Will eventually provide pixel, interpolation, jitter, atmospheric, and
other MTF functions for use in general-purpose airborne sensor calculations.

"""

# python 3 compatibility
from __future__ import division			# single / does float div
from __future__ import print_function 	# watch out for 2.X compatibility

# other modules
import math
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
import matplotlib.colors as cm


# ---------------------------------------------
# Functions for optical MTF
# ---------------------------------------------

def mtf_circ(sf, wave=0.000532, fno=5):

	"""
	Diffraction-limited circular pupil MTF function (no obscuration)
	Takes spatial frequency in cy/mm, wavelength in mm, and f/# as arguments
	- Wavelength defaults to 532 nm.
	- f/# defaults to f/5
	"""

	sfcut = 1 / (wave * float(fno)) 	# define optical cutoff
	nsf = sf / sfcut 					# normalize
	if nsf < 0 or nsf > 1:				# keep frequency in bounds
		mtf = 0
	elif nsf == 0:
		mtf = 1
	else:
		mtf = (2/3.14159)*(math.acos(nsf) - nsf * math.sqrt(1 - nsf**2))
	return mtf

# ---------------

def mtf_blur(sf):

	"""
	Aberration MTF function
	Takes spatial frequency in cy/mm, wavelength in mm, and f/# as arguments
	- Wavelength defaults to 532 nm.
	- f/# defaults to f/5
	"""
	return 1

def mtf_obsc(sf):

	"""
	Obscured diffraction-limited MTF function
	Takes spatial frequency in cy/mm, wavelength in mm, and f/# as arguments
	- Wavelength defaults to 532 nm.
	- f/# defaults to f/5
	"""
	return 1

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# Test code
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------


if __name__ == '__main__':

	print
	print('-'*75)

	# set sensor parameters
	baseSF		= 	100	# cy/mm
	baseWave	=	0.000532	# mm
	baseFno		=	5.4
	fcut = 1/(baseWave * float(baseFno))
	numMTFpts	=	350
	
	# run a single calculation
	m1			=	mtf_circ(baseSF, baseWave, baseFno)
	print(
	'The MTF at %3.3f cy/mm for %1.3f um at f/%1.1f is %1.15f' % (
		baseSF, baseWave * 1000, baseFno, m1
	))

	# now a table of calculations
	print('\n')
	print('Now generate a table of MTF values:')
	print('SF\tMTF')
	sfList = np.arange(0.0, fcut, fcut/numMTFpts)	# make it a numpy array
	m2List = []
	i = 0
	for sf in sfList:
		i += 1
		m2 = mtf_circ(sf, baseWave, baseFno)
		m2List.append(m2)
		if not i % 20:	# print only every 20th line...
			print(
			#TODO right justify columnar numbers
			  '%3.2f' % sf,
			  '\t',
			  '%1.3f'.rjust(5) % m2
			  )

	# plt.plot(sfList, m2List)
	# plt.show()

	# now integrate the table in various ways
	print('\n')
	print('Now integrate this MTF out to %3.3f cy/mm:' % sfList[-1] )
	
	integ1 = integrate.trapz(np.array(m2List), sfList)
	integ2 = integrate.simps(np.array(m2List), dx = fcut/numMTFpts)
	print(
		'Method 1: scipy.trapz()\t',integ1,
		'\nMethod 2: scipy.simps()\t',integ2)
	print('\n')

	# now attempt to integrate edge response function with sine kernel
	pixelPos = 0.5 			# relative pixel position
	pixelSize = 0.00454 	# physical pixel pitch in mm
	
	# TODO: make duplicate lists into list or tuple pairs
	# TODO: make into general class-compliant function eventually
	sinList1 = []
	sinList2 = []
	for sf in sfList:
		if sf == 0:
			sn1 = 0
			sn2 = 0
		else:
			sn1 = math.sin(2 * 3.14159 * sf * pixelSize * pixelPos) / sf
			sn2 = math.sin(2 * 3.14159 * sf * pixelSize * (-pixelPos)) / sf

		sinList1.append(sn1)
		sinList2.append(sn2)

	integrand1 = np.array(m2List) * np.array(sinList1)
	integrand2 = np.array(m2List) * np.array(sinList2)

	# print(integrand)
	integ3a = 0.5 + (1 / 3.14159) * integrate.simps(integrand1, sfList)
	integ3b = 0.5 + (1 / 3.14159) * integrate.simps(integrand2, sfList)
	rer = integ3a - integ3b
	print('For pixel size %2.2f um at f/%3.2f' % (1000.0 * pixelSize, baseFno) ),
	# print('at f/%3.2f' % baseFno)
	print('Edge response integrals:\t\t%f, %f' % (integ3a, integ3b)) 
	print('and relative edge response (RER):\t%f' % rer)
	print('\n')

	print('Done for now...\n')
	print('-'*75)

	p1 = plt.plot(sfList,m2List)

#	plt.xlabel('X position (urad)')
#	plt.ylabel('Y position (urad)')
#	cb = plt.colorbar(p1)
#	cb.set_label("Normalized Grayscale Intensity Value")
	plt.show()

