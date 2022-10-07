# Author: John Hennin
# Last Updated: 2020-11-05

# Script Description:
# This script reads all GeoTIFF files in the local directory and produces new ASCII raster files with the same data, as well as complementary PRJ files
# It also generates a text file describing the amount of each unique cell value in each raster file. This can be used for analysis purposes.

import rasterio
import numpy as np
import argparse
import os

if __name__ == "__main__":
    parser= argparse.ArgumentParser(description="This script reads all GeoTIFF files in the local directory and produces new ASCII raster files with the same data, as well as complementary PRJ files.\nIt also generates a text file describing the amount of each unique cell value in each raster file. This can be used for analysis purposes.")
    args = parser.parse_args()

    files = []
    for file in os.listdir("."):
        # Check for tif files in local directory
        if file[-3:] == "tif":
            files.append(file)
            with rasterio.open(file) as tif:

                # Save ASCII raster header and PRJ file features
                arraydata = tif.read(tif.indexes[0])
                ncols = tif.width
                nrows = tif.height
                crs = tif.crs
                xllcorner, yllcorner = tif.transform * (0,tif.height)
                cellsize = tif.transform[0]
                prjinfo = crs.wkt

            # Write header information and data in new ASCII raster file of same name as original GeoTIFF
            with open(file[:-4]+".asc","w+") as ascfile:
                ascfile.write("ncols " + str(ncols) + '\n' + "nrows " + str(nrows) + '\n' + "xllcorner " + str(xllcorner) + '\n' + "yllcorner " + str(yllcorner) + '\n' + "cellsize " + str(cellsize) + '\n')
                for line in arraydata:
                    for pixel in line:
                        ascfile.write(str(pixel) + " ")
                    ascfile.write("\n")

            # Write PRJ information in new PRJ file of same name as original GeoTIFF
            with open(file[:-4]+".prj","w+") as prjfile:
                prjfile.write(prjinfo)

            # Save count of cells per each unique value in raster file
            valcounts = []
            for value in np.unique(arraydata):
                valcount = 0
                for line in arraydata:
                    for pixel in line:
                        if pixel == value:
                            valcount += 1
                valcounts.append(valcount)

            # Create text file with first file
            if file == files[0]:
                with open("cell_counts.txt","w+") as countfile:
                    countfile.write(str(file[:-4])+" Unique Values and Amounts:\n")
                    for i in np.arange(len(np.unique(arraydata))):
                        countfile.write(str(np.unique(arraydata)[i])+" : "+str(valcounts[i])+"\n")
                        if i == len(np.unique(arraydata))-1:
                            countfile.write("\n")

            # Add to text file if more than one file converted
            else:
                with open("cell_counts.txt","a") as countfile:
                    countfile.write(str(file[:-4])+" Unique Values and Amounts:\n")
                    for i in np.arange(len(np.unique(arraydata))):
                        countfile.write(str(np.unique(arraydata)[i])+" : "+str(valcounts[i])+"\n")
                        if i == len(np.unique(arraydata))-1:
                            countfile.write("\n")
