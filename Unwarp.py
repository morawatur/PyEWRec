import numpy as np
import ImageSupport as imsup
import CrossCorr as cc
import Propagation as prop
from skimage import transform as tf

# -------------------------------------------------------------------

# zakladam, ze obrazy sa wstepnie zsuniete
def UnwarpImage(imgRef, img, nDiv):
    mt = img.memType
    dt = img.cmpRepr

    dfChange = -abs(abs(imgRef.defocus) - abs(img.defocus))
    print('df_uw({0}, {1}) = {2:.2f} um'.format(imgRef.numInSeries, img.numInSeries, dfChange * 1e6))
    imgRefProp = prop.PropagateBackToDefocus(imgRef, dfChange)

    fragCoords = [(b, a) for a in range(nDiv) for b in range(nDiv)]
    shifts = cc.CalcPartialCrossCorrFunUW(imgRefProp, img, nDiv, fragCoords)

    fName = 'uw_out.txt'
    textFile = open(fName, 'ab')
    np.savetxt(textFile, shifts, fmt='%.1f')
    textFile.write(b'---------------------------------------------------------------\n')
    textFile.close()

    fragDimSize = img.width // nDiv
    src = np.array(fragCoords)
    src *= fragDimSize
    dst = src - shifts

    img.ReIm2AmPh()
    img.MoveToCPU()
    oldMin, oldMax = np.min(img.amPh.am), np.max(img.amPh.am)
    scaledArray = imsup.ScaleImage(img.amPh.am, -1.0, 1.0)

    tform3 = tf.ProjectiveTransform()
    tform3.estimate(src, dst)
    warped = tf.warp(scaledArray, tform3, output_shape=(img.height, img.width)).astype(np.float32)
    warpedScaledBack = imsup.ScaleImage(warped, oldMin, oldMax)

    warpedImage = imsup.Image(warped.shape[0], warped.shape[1])
    warpedImage.amPh.am = np.copy(warpedScaledBack)
    img.amPh.am = np.copy(warpedImage.amPh.am)

    img.ChangeMemoryType(mt)
    img.ChangeComplexRepr(dt)
    imgRef.ChangeMemoryType(mt)
    imgRef.ChangeComplexRepr(dt)

    return img

# -------------------------------------------------------------------

def UnwarpImageList(imgList, nDiv):
    # imgListCopy = imsup.ImageList()
    # for img in imgList:
    #    imgListCopy.append(imsup.CopyImage(img))
    # for img, imgRef, idx in zip(imgList[1:], imgListCopy[:len(imgListCopy) - 1], range(1, len(imgList))):

    # ---
    imgRef = imgList[0]
    imgRef2 = imsup.CopyImage(imgList[10])
    for img, idx in zip(imgList[1:11], range(1, 11)):
        df = img.defocus
        imgList[idx] = UnwarpImage(imgRef, img, nDiv)
        imgList[idx].defocus = df
        warpPath = 'results/warp/imgw{0}.png'.format(img.numInSeries)
        imsup.SaveAmpImage(imgList[idx], warpPath)

    for img, idx in zip(imgList[11:], range(11, len(imgList))):
        df = img.defocus
        imgList[idx] = UnwarpImage(imgRef2, img, nDiv)
        imgList[idx].defocus = df
        warpPath = 'results/warp/imgw{0}.png'.format(img.numInSeries)
        imsup.SaveAmpImage(imgList[idx], warpPath)
    # ---

    # imgList.reverse()
    # imgRef = imgList[0]
    # for img, idx in zip(imgList[1:], range(1, len(imgList))):
    #     df = img.defocus
    #     imgList[idx] = UnwarpImage(imgRef, img, nDiv)
    #     imgList[idx].defocus = df
    #     warpPath = 'results/warpNew/imgw{0}.png'.format(img.numInSeries)
    #     imsup.SaveAmpImage(imgList[idx], warpPath)
    # imgList.reverse()
    return imgList
