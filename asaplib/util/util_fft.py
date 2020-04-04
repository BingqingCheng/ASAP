"""
Functions for performing Fast Fourier transforms
"""

import numpy as np

"""
# Fourier analysis of x(t) to obtain $x(\omega)$.
## $FT(x(t)) = x(\omega)$
## $S_x(\omega)=FT(<x(0)x(t)>) = x(\omega) \cdot x(\omega) $

#  In reality we have a list $\{ x_{1...n}\}$
## $A_k = \sum_{m=0}^{n-1} x_m \exp(-2\pi i \dfrac{m k }{n})$
## $x(\omega = \dfrac{2\pi}{n\Delta t}k) = \Delta t A_k$ 

# Fourier analysis of v(t) to obtain $v(\omega)$.
## $FFT(v(t)) = v(\omega)=i\omega x(\omega)$
## $FFT(<x(0)v(t)>) = -x(\omega) \cdot v(\omega) $
##  $FFT(<x(0)v(t)>) = i \omega x(\omega) \cdot x(\omega) $
## $FFT(<v(0)v(t)>) = v(\omega) \cdot v(\omega) $
##  $FFT(<v(0)v(t)>) = -(i \omega)^2 x(\omega) \cdot x(\omega) = \omega^2 x(\omega) \cdot x(\omega)$

EXAMPLES:
testfx = fftranform(txv[:,[0,1]],len(txv)) fftranform(txv[:,[0,1]],len(txv)) 
testreversefx = ifftranform(testfx) 

testftcxx = rfftcrosscorr(txv[:,[0,1]],txv[:,[0,1]],10000)
"""


def smooth(y, box_pts):
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


def fftranform(x, dlen=10000):
    # input: a list of real signals {x} with timestep dt
    # output: x(omega)

    # make sure that we have odd number of signals as it makes fft easier
    if dlen % 2 == 0:
        dlen -= 1
        # x.append(x[0])
    # this is the FFT expansion coeficients assuming inputs are real numbers
    # https://docs.scipy.org/doc/numpy-1.13.0/reference/routines.fft.html#module-numpy.fft
    dt = x[1, 0] - x[0, 0]  # assume the timestep is constant
    window = len(x) // dlen
    # print dlen
    omega0 = 2.0 * np.pi / (dlen - 1) / dt
    xomega = np.zeros((dlen, 2), dtype=np.complex_)
    xomega[0:dlen // 2 + 1, 0] = np.arange(dlen // 2 + 1) * omega0
    xomega[dlen // 2 + 1:, 0] = np.arange(dlen // 2, 0, -1) * omega0 * -1

    for i in range(window):
        dx = x[i * dlen:(i + 1) * dlen, 1]
        Ax = np.fft.fft(dx[:], axis=0)
        # print len(Ax)
        xomega[:, 1] += Ax * dt
    xomega[:, 1] /= window
    # return [omega, A(omega)]
    """ when numpy does fft, A[1:n/2] contains the positive-frequency terms, 
    and A[n/2+1:] contains the negative-frequency terms, 
    in order of decreasingly negative frequency.
    """
    return xomega


def ifftranform(xomega):
    # first retrieve the descrete rFFT coefficients
    # note that numpy uses a normalization factor of 1/n here but not during the forward fft
    omega0 = xomega[1, 0] - xomega[0, 0]
    dt = np.pi / (len(xomega) // 2) / omega0
    Ax = xomega[:, 1] / dt
    x = np.fft.ifft(Ax[:], axis=0)
    return np.column_stack((dt * np.arange(len(x)), x))


def fftcrosscorr(x, y, dlen=10000):
    # make sure that we have odd number of signals as it makes fft easier
    if dlen % 2 == 0:
        dlen -= 1
        # x.append(x[0])
        # y.append(y[0])

    # the fft coecofficents of the crosscorrelation function c_xy(t)
    dt = x[1, 0] - x[0, 0]  # assume the timestep is constant
    window = len(x) // dlen
    omega0 = 2.0 * np.pi / (dlen - 1) / dt
    cxyomega = np.zeros((dlen, 2), dtype=np.complex_)
    cxyomega[0:dlen // 2 + 1, 0] = np.arange(dlen // 2 + 1) * omega0
    cxyomega[dlen // 2 + 1:, 0] = np.arange(dlen // 2, 0, -1) * omega0 * -1

    for i in range(window):
        dx = x[i * dlen:(i + 1) * dlen, 1]
        dy = y[i * dlen:(i + 1) * dlen, 1]
        Ax = np.fft.fft(dx[:], axis=0)
        Ay = np.fft.fft(dy[:], axis=0)

        cxyomega[:, 1] += np.conjugate(Ax[:]) * Ay[:] / dlen * dt

    for i in range(window - 1):
        dx = x[i * dlen + dlen // 2:(i + 1) * dlen + dlen // 2, 1]
        dy = y[i * dlen + dlen // 2:(i + 1) * dlen + dlen // 2, 1]
        Ax = np.fft.fft(dx[:], axis=0)
        Ay = np.fft.fft(dy[:], axis=0)

        cxyomega[:, 1] += np.conjugate(Ax[:]) * Ay[:] / dlen * dt

    cxyomega[:, 1] /= (window * 2 - 1)
    return cxyomega


def rfftranform(x, dlen=10000):
    # input: a list of real signals {x} with timestep dt
    # output: x(omega)

    # this is the FFT expansion coeficients assuming inputs are real numbers
    # https://docs.scipy.org/doc/numpy-1.13.0/reference/routines.fft.html#module-numpy.fft
    dt = x[1, 0] - x[0, 0]  # assume the timestep is constant
    window = len(x) // dlen
    # print dlen
    omega0 = 2.0 * np.pi / dlen / dt
    xomega = np.zeros((dlen // 2 + 1, 2), dtype=np.complex_)
    xomega[:, 0] = np.arange(dlen // 2 + 1) * omega0

    for i in range(window):
        dx = x[i * dlen:(i + 1) * dlen, 1]
        Ax = np.fft.rfft(dx[:], axis=0)
        # print len(Ax)
        xomega[:, 1] += Ax * dt
    xomega[:, 1] /= window
    # return [omega, A(omega)]
    """ when numpy does fft, A[1:n/2] contains the positive-frequency terms, 
    and A[n/2+1:] contains the negative-frequency terms, 
    in order of decreasingly negative frequency.
    Hoever, When the DFT is computed for purely real input, 
    the output is Hermitian-symmetric, 
    i.e. the negative frequency terms are just the complex conjugates of the corresponding positive-frequency terms, 
    and the negative-frequency terms are therefore redundant. 
    This function does not compute the negative frequency terms, 
    and the length of the transformed axis of the output is therefore n//2 + 1
    """
    return xomega


def irfftranform(xomega):
    # first retrieve the descrete rFFT coefficients
    # note that numpy uses a normalization factor of 1/n here but not during the forward fft
    omega0 = xomega[1, 0] - xomega[0, 0]
    dt = np.pi / len(xomega) / omega0
    Ax = xomega[:, 1] / dt
    x = np.fft.irfft(Ax[:], axis=0)
    return np.column_stack((dt * np.arange(len(x)), x))


def rfftcrosscorr(x, y, dlen=10000):
    # the fft coecofficents of the crosscorrelation function c_xy(t)
    dt = x[1, 0] - x[0, 0]  # assume the timestep is constant
    window = len(x) // dlen
    omega0 = 2.0 * np.pi / dlen / dt
    cxyomega = np.zeros((dlen // 2 + 1, 2), dtype=np.complex_)
    cxyomega[:, 0] = np.arange(dlen // 2 + 1) * omega0

    for i in range(window):
        dx = x[i * dlen:(i + 1) * dlen, 1]
        dy = y[i * dlen:(i + 1) * dlen, 1]
        Ax = np.fft.rfft(dx[:], axis=0)
        Ay = np.fft.rfft(dy[:], axis=0)

        cxyomega[:, 1] += np.conjugate(Ax[:]) * Ay[:] / dlen * dt

    for i in range(window - 1):
        dx = x[i * dlen + dlen // 2:(i + 1) * dlen + dlen // 2, 1]
        dy = y[i * dlen + dlen // 2:(i + 1) * dlen + dlen // 2, 1]
        Ax = np.fft.rfft(dx[:], axis=0)
        Ay = np.fft.rfft(dy[:], axis=0)

        cxyomega[:, 1] += np.conjugate(Ax[:]) * Ay[:] / dlen * dt

    cxyomega[:, 1] /= (window * 2 - 1)
    return cxyomega
