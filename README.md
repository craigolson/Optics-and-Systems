# Optics-and-Systems
Class lineofsight.py

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

- LeMaster, D. and Eismann, M., ìpyBSM: A Python package for modeling imaging 
systems,î Proc. SPIE 10204; Long-Range Imaging II, 1020405 (2017), 
doi: 10.1117/12.2262561

The calculation methods are based entirely on openly available formalism primarily 
fromthe following two sources:

- Kopeika, N., [A System Approach to Imaging], SPIE Press, (1998).

- Youngworth, R., Gallagher, B.B., and Stamper, B.L., ìAn overview of power 
spectral density (PSD) calculations,î Proc. SPIE 5869; Optical Manufacturing 
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
