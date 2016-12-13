# Nanoparticle sizing tool

Simple software tool to segment images into nanoparticles, and find
nanoparticle size statistics. Originally developed for Eng. 241 Autumn
2016 at Stanford University.

## Installation and dependencies

Runs on Python 2 (not 3).

Package dependencies: numpy, scipy, matplotlib, scikit-image

## How to run

### Formatting the CSV file

Create a CSV file containing input information. This CSV file provides
file paths and pixel-to-length information. The CSV must be structured in
the following way

#### First line (header)

``` Histogram file path (where to save histogram) [str], Histogram title
[str], histogram width in inches [number], histogram height in inches
[number], histogram bin width in microns [number], histogram max bin to
display in microns [number] ```

#### All other lines after the first (body)

``` Absolute file path to SEM image [string], number of microns per pixel
[number], density [string HD or LD]```

HD selects the high particle density algorithm, which is better for dense
particles (i.e. the entire image is packed with particles).

LD selects the low particle density algorithm, which is better for very
sparse particles (i.e. very small number of particles, most of the SEM
frame is empty space).

An example CSV file is given in the repository.

### Running the tool

Run `nanoparticlesize.py` with the CSV file as the only positional
argument, i.e.

```
python nanoparticlesize.py mycsv.csv
```


The program will print out the mean and median particle sizes in microns.
The program will also generate two PDF files: (1) a histogram showing particle
diameter distribution; (2) segmented version of the original TIF, showing
how the program segmented the image into particles. 

For help, run

```
python nanoparticlesize.py -h
```

## Sample output

Run the command
```
    python nanoparticlesize.py sample.csv
```

You should see the output

```
0.12226612117 um median size
0.13292668663 um mean size
```

The sample histogram will be in `sample_histogram.pdf` and the sample
segmented image will be in `segmented_sample.pdf`.

## How does it work?

For images with low density particles (LD mode), the primary challenge is
isolating particles from the background. Assuming that the background is
flat and smooth, low density particles are found as follows:

1. Threshold with Otsu thresholding
2. Calculate distance transform to edges
3. Find local maxima in distance-transformed image, 21x21 pixel boxes
4. Start watershed algorithm from those peaks, each filled-in region
   corresponds to a particle

For images with high density particles (HD mode), the primary challenge is
separating particles that are adjacent to each other. Edge detection
therefore becomes a higher priority, and high density particles are found
as follows:

1. Sobel edge detection, and normalization of edges
2. Isodata thresholding of edges
3. Calculate distance transform to edges
3. Find local maxima in distance-transformed image, 21x21 pixel boxes
4. Start watershed algorithm from these peaks, each filled-in region
   corresponds to a particle

## TODO

* Image processing / metadata processing to compute pixel-to-length scale
  from scale bar automatically. Challenge is doing this consistently for
  all files...
* Automated removal of scale at bottom of image. Currently hardcoded,
  works for FEI Sirion @ SNSF.
