// Mean and fraction of saturated (>= 253 of 255, i.e. >= 0.99) and dark (<= 2, i.e. <= 0.01) pixels
// of the current 8-bit image. Run by src/fiji_check.py, or by hand from Plugins > Macros > Run.
#@ ImagePlus image
#@output String mean
#@output String sat
#@output String dark
selectImage(image);
getStatistics(area, m, min, max, std, histogram);
n = getWidth() * getHeight();
s = 0; d = 0;
for (i = 253; i < 256; i++) s += histogram[i];
for (i = 0; i <= 2; i++) d += histogram[i];
mean = "" + m;
sat = "" + (s / n);
dark = "" + (d / n);
