import numpy as np
from numba import cuda
import math
import Constants as const
import CudaConfig as ccfg
import Array as carr
import ImageSupport as cimsup
import CrossCorr as ccc

# -------------------------------------------------------------------

def CalcTransferFunction(imgDim, pxDim, defocusChange):
    blockDim, gridDim = ccfg.DetermineCudaConfig(imgDim)
    ctf = cimsup.Image(imgDim, imgDim, cimsup.Image.cmp['CAP'], cimsup.Image.mem['GPU'])
    ctfCoeff = np.pi * const.ewfLambda * defocusChange
    CalcTransferFunction_dev[gridDim, blockDim](ctf.amPh.am, ctf.amPh.ph, imgDim, pxDim, ctfCoeff)
    ctf.defocus = defocusChange
    return ctf

# -------------------------------------------------------------------

@cuda.jit('void(float32[:, :], int32, int32)')
def CalcRecSquareDistances_dev(rsd, imgDim, pxDim):
    x, y = cuda.grid(2)
    if x >= imgDim or y >= imgDim:
        return
    recPxWidth = 1.0 / (imgDim * pxDim)
    recOrigin = -1.0 / (2.0 * pxDim)
    recXDist = recOrigin + x * recPxWidth
    recYDist = recOrigin + y * recPxWidth
    rsd[x, y] = recXDist * recXDist + recYDist * recYDist

# -------------------------------------------------------------------

@cuda.jit('void(float32[:, :], float32[:, :], int32, float32, float32)')
def CalcTransferFunction_dev(ctfAm, ctfPh, imgDim, pxDim, ctfCoeff):
    x, y = cuda.grid(2)
    if x >= imgDim or y >= imgDim:
        return

    recPxWidth = 1.0 / (imgDim * pxDim)
    recOrigin = -1.0 / (2.0 * pxDim)

    recXDist = recOrigin + x * recPxWidth
    recYDist = recOrigin + y * recPxWidth
    recSquareDist = recXDist * recXDist + recYDist * recYDist

    ctfAm[x, y] = 1.0
    ctfPh[x, y] = ctfCoeff * recSquareDist

# -------------------------------------------------------------------

def PropagateWave(img, ctf):
    fft = ccc.FFT(img)
    fft.ReIm2AmPh()

    ctf = ccc.Diff2FFT(ctf)     # !!!
    ctf.ReIm2AmPh()

    fftProp = cimsup.Image(img.height, img.width, cimsup.Image.cmp['CAP'], cimsup.Image.mem['GPU'])
    fftProp.amPh = cimsup.MultAmPhMatrices(fft.amPh, ctf.amPh)

    imgProp = ccc.IFFT(fftProp)
    imgProp.ReIm2AmPh()
    imgProp.defocus = img.defocus + ctf.defocus
    return imgProp

# -------------------------------------------------------------------

def PropagateToFocus(img):
    ctf = CalcTransferFunction(img.width, img.pxWidth, -img.defocus)
    return PropagateWave(img, ctf)

# -------------------------------------------------------------------

def PropagateBackToDefocus(img, defocus):
    ctf = CalcTransferFunction(img.width, img.pxWidth, defocus)
    # img.defocus = 0.0
    return PropagateWave(img, ctf)

# -------------------------------------------------------------------

def PerformIWFR(images, N):
    imgHeight, imgWidth = images[0].height, images[0].width
    ewfResultsDir = 'results/ewf/'
    ewfAmName = 'am'
    ewfPhName = 'ph'
    backCTFunctions = []
    forwCTFunctions = []

    print('Starting IWFR...')

    # storing contrast transfer functions

    for img in images:
        # print(imgWidth, img.pxWidth, -img.defocus)
        ctf = CalcTransferFunction(imgWidth, img.pxWidth, -img.defocus)
        ctf.AmPh2ReIm()
        # ctf.reIm = ccc.Diff2FFT(ctf.reIm)
        backCTFunctions.append(ctf)

        ctf = CalcTransferFunction(imgWidth, img.pxWidth, img.defocus)
        ctf.AmPh2ReIm()
        # ctf.reIm = ccc.Diff2FFT(ctf.reIm)
        forwCTFunctions.append(ctf)

    # start IWFR procedure

    exitWave = cimsup.Image(imgHeight, imgWidth, cimsup.Image.cmp['CRI'], cimsup.Image.mem['GPU'])

    for i in range(0, N):
        print('Iteration no {0}...'.format(i+1))
        cimsup.ClearImageData(exitWave)

        # backpropagation (to in-focus plane)

        for img, idx in zip(images, range(0, len(images))):
            img = PropagateWave(img, backCTFunctions[idx])
            # img = PropagateToFocus(img)
            img.AmPh2ReIm()
            exitWave.reIm = carr.AddTwoArrays(exitWave.reIm, img.reIm)

        exitWave.reIm = carr.MultArrayByScalar(exitWave.reIm, 1/len(images))

        ewfAmPath = ewfResultsDir + ewfAmName + str(i+1) + '.png'
        ewfPhPath = ewfAmPath.replace(ewfAmName, ewfPhName)
        cimsup.SaveAmpImage(exitWave, ewfAmPath)
        cimsup.SavePhaseImage(exitWave, ewfPhPath)

        # forward propagation (to the original focus plane)

        for img, idx in zip(images, range(0, len(images))):
            images[idx] = PropagateWave(exitWave, forwCTFunctions[idx])
            # images[idx] = PropagateBackToDefocus(exitWave, img.defocus)
            images[idx].amPh.am = img.amPh.am           # restore original amplitude

    print('All done')

    return exitWave