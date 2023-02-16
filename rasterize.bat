@echo off
mkdir %1_label
cd %1
for %%i in (*.tif) do (
echo Working on %%i...
gdal_create -if %%i ../%1_label/%%i -burn 0 -bands 1 -ot Byte
gdal_rasterize -burn 1 -l %3 ../%2 ../%1_label/%%i
)
cd ..
gdalbuildvrt %1_label.vrt %1_label/*.tif -resolution highest
gdalbuildvrt %1.vrt %1/*.tif -resolution highest