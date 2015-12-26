# basically reformatted code from here: https://github.com/TimSC/image-piecewise-affine retooled to work without PIL

import numpy
import scipy.spatial as spatial


def get_bilinear_pixel(imArr, posX, posY, out):
    # Get integer and fractional parts of numbers
    modXi = int(posX)
    modYi = int(posY)
    modXf = posX - modXi
    modYf = posY - modYi

    # Get pixels in four corners
    for chan in range(imArr.shape[2]):
        bl = imArr[modYi, modXi, chan]
        br = imArr[modYi, modXi + 1, chan]
        tl = imArr[modYi + 1, modXi, chan]
        tr = imArr[modYi + 1, modXi + 1, chan]

        # Calculate interpolation
        b = modXf * br + (1. - modXf) * bl
        t = modXf * tr + (1. - modXf) * tl
        pxf = modYf * t + (1. - modYf) * b
        out[chan] = int(pxf + 0.5)  # Do fast rounding to integer

    return None  # Helps with profiling view


def warp_process(inIm, inArr,
                 outArr,
                 inTriangle,
                 triAffines, shape):
    # Ensure images are 3D arrays
    px = numpy.empty((inArr.shape[2],), dtype=numpy.int32)
    homogCoord = numpy.ones((3,), dtype=numpy.float32)

    # Calculate ROI in target image
    xmin = shape[:, 0].min()
    xmax = shape[:, 0].max()
    ymin = shape[:, 1].min()
    ymax = shape[:, 1].max()
    xmini = int(xmin)
    xmaxi = int(xmax + 1.)
    ymini = int(ymin)
    ymaxi = int(ymax + 1.)
    # print xmin, xmax, ymin, ymax

    # Synthesis shape norm image
    for i in range(xmini, xmaxi):
        for j in range(ymini, ymaxi):
            homogCoord[0] = i
            homogCoord[1] = j

            # Determine which tesselation triangle contains each pixel in the shape norm image
            if i < 0 or i >= outArr.shape[1]: continue
            if j < 0 or j >= outArr.shape[0]: continue

            # Determine which triangle the destination pixel occupies
            tri = inTriangle[i, j]
            if tri == -1:
                continue

            # Calculate position in the input image
            affine = triAffines[tri]
            outImgCoord = numpy.dot(affine, homogCoord)

            # Check destination pixel is within the image
            if outImgCoord[0] < 0 or outImgCoord[0] >= inArr.shape[1]:
                for chan in range(px.shape[0]): outArr[j, i, chan] = 0
                continue
            if outImgCoord[1] < 0 or outImgCoord[1] >= inArr.shape[0]:
                for chan in range(px.shape[0]): outArr[j, i, chan] = 0
                continue

            # Nearest neighbour
            # outImgL[i,j] = inImgL[int(round(inImgCoord[0])),int(round(inImgCoord[1]))]

            # Copy pixel from source to destination by bilinear sampling
            # print i,j,outImgCoord[0:2],im.size
            get_bilinear_pixel(inArr, outImgCoord[0], outImgCoord[1], px)
            for chan in range(px.shape[0]):
                outArr[j, i, chan] = px[chan]
                # print outImgL[i,j]

    return None


def piecewise_affine(srcIm, srcPoints, dstIm, dstPoints):
    # Convert input to correct types
    srcArr = numpy.asarray(srcIm, dtype=numpy.float32)
    dstPoints = numpy.array(dstPoints)
    srcPoints = numpy.array(srcPoints)

    # Split input shape into mesh
    tess = spatial.Delaunay(dstPoints)

    # Calculate ROI in target image
    xmin, xmax = dstPoints[:, 0].min(), dstPoints[:, 0].max()
    ymin, ymax = dstPoints[:, 1].min(), dstPoints[:, 1].max()
    # print xmin, xmax, ymin, ymax

    # Determine which tesselation triangle contains each pixel in the shape norm image
    inTessTriangle = numpy.ones(srcIm.shape[:2], dtype=numpy.int) * -1
    for i in range(int(xmin), int(xmax + 1.)):
        for j in range(int(ymin), int(ymax + 1.)):
            if i < 0 or i >= inTessTriangle.shape[0]: continue
            if j < 0 or j >= inTessTriangle.shape[1]: continue
            normSpaceCoord = (float(i), float(j))
            simp = tess.find_simplex([normSpaceCoord])
            inTessTriangle[i, j] = simp

    # Find affine mapping from input positions to mean shape
    triAffines = []
    for i, tri in enumerate(tess.vertices):
        meanVertPos = numpy.hstack((srcPoints[tri], numpy.ones((3, 1)))).transpose()
        shapeVertPos = numpy.hstack((dstPoints[tri, :], numpy.ones((3, 1)))).transpose()

        affine = numpy.dot(meanVertPos, numpy.linalg.inv(shapeVertPos))
        triAffines.append(affine)

    # Prepare arrays, check they are 3D
    targetArr = numpy.copy(numpy.asarray(dstIm, dtype=numpy.uint8))
    srcArr = srcArr.reshape(srcArr.shape[0], srcArr.shape[1], srcIm.shape[2])
    targetArr = targetArr.reshape(targetArr.shape[0], targetArr.shape[1], srcIm.shape[2])

    triAffines = numpy.asarray(triAffines)
    # Calculate pixel colours
    warp_process(srcIm, srcArr, targetArr, inTessTriangle, triAffines, dstPoints)

    # Convert single channel images to 2D
    if targetArr.shape[2] == 1:
        targetArr = targetArr.reshape((targetArr.shape[0], targetArr.shape[1]))
    return targetArr
