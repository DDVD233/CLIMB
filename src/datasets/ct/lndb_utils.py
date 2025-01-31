import csv

import numpy as np
import SimpleITK as sitk
from scipy.ndimage import zoom


def readCsv(csvfname):
    # read csv to list of lists
    with open(csvfname, 'r') as csvf:
        reader = csv.reader(csvf)
        csvlines = list(reader)
    return csvlines


def writeCsv(csfname, rows):
    # write csv from list of lists
    with open(csfname, 'w', newline='') as csvf:
        filewriter = csv.writer(csvf)
        filewriter.writerows(rows)


def readMhd(filename):
    # read mhd/raw image
    itkimage = sitk.ReadImage(filename)
    scan = sitk.GetArrayFromImage(itkimage)  #3D image
    #  print(scan.shape)

    spacing = itkimage.GetSpacing()  #voxelsize
    origin = itkimage.GetOrigin()  #world coordinates of origin
    transfmat = itkimage.GetDirection()  #3D rotation matrix
    #  print(dir(itkimage))
    return scan, spacing, origin, transfmat


def writeMhd(filename, scan, spacing, origin, transfmat):
    # write mhd/raw image
    itkim = sitk.GetImageFromArray(scan, isVector=False)  #3D image
    itkim.SetSpacing(spacing)  #voxelsize
    itkim.SetOrigin(origin)  #world coordinates of origin
    itkim.SetDirection(transfmat)  #3D rotation matrix
    sitk.WriteImage(itkim, filename, False)


def getImgWorldTransfMats(spacing, transfmat):
    # calc image to world to image transformation matrixes
    transfmat = np.array([transfmat[0:3], transfmat[3:6], transfmat[6:9]])
    for d in range(3):
        transfmat[0:3, d] = transfmat[0:3, d] * spacing[d]
    transfmat_toworld = transfmat  #image to world coordinates conversion matrix
    transfmat_toimg = np.linalg.inv(transfmat)  #world to image coordinates conversion matrix

    return transfmat_toimg, transfmat_toworld


def convertToImgCoord(xyz, origin, transfmat_toimg):
    # convert world to image coordinates
    xyz = xyz - origin
    xyz = np.round(np.matmul(transfmat_toimg, xyz))
    return xyz


def convertToWorldCoord(xyz, origin, transfmat_toworld):
    # convert image to world coordinates
    xyz = np.matmul(transfmat_toworld, xyz)
    xyz = xyz + origin
    return xyz


def extractCube(scan, spacing, xyz, cube_size=80, cube_size_mm=51):
    # Extract cube of cube_size^3 voxels and world dimensions of cube_size_mm^3 mm from scan at image coordinates xyz
    xyz = np.array([xyz[i] for i in [2, 1, 0]], int)
    spacing = np.array([spacing[i] for i in [2, 1, 0]])
    scan_halfcube_size = np.array(cube_size_mm / spacing / 2, int)
    if np.any(xyz < scan_halfcube_size) or np.any(xyz + scan_halfcube_size > scan.shape):  # check if padding is necessary
        maxsize = max(scan_halfcube_size)
        scan = np.pad(
            scan, ((
                maxsize,
                maxsize,
            )), 'constant', constant_values=0
        )
        xyz = xyz + maxsize

    scancube = scan[xyz[0] - scan_halfcube_size[0]:xyz[0] + scan_halfcube_size[0],  # extract cube from scan at xyz
                    xyz[1] - scan_halfcube_size[1]:xyz[1] + scan_halfcube_size[1],
                    xyz[2] - scan_halfcube_size[2]:xyz[2] + scan_halfcube_size[2]]

    sh = scancube.shape

    scancube = zoom(scancube, (cube_size / sh[0], cube_size / sh[1], cube_size / sh[2]), order=2)  #resample for cube_size

    return scancube