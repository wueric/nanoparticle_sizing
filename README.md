# Nanoparticle sizing tool

## Installation and dependencies

Runs on Python2 (not 3)

Package dependencies: numpy, scipy, matplotlib, scikit-image

## How to run

Create a CSV file containing input information. This CSV file provides
file paths and pixel-to-length information. The CSV must be structured in
the following way:

First line (header) must have the format:

Histogram file path (where to save histogram) [str], Histogram title
[str], histogram width in inches [number], histogram height in inches
[number], histogram bin width in microns [number], histogram max bin to
display in microns [number]

All other lines after the first have the format:

Absolute file path to SEM image [string], number of microns per pixel
[number], density [string HD or LD]

An example CSV file is given in the repository.

Run `nanoparticlesize.py` with the CSV file as the only positional
argument, i.e.

```
python nanoparticlesize.py mycsv.csv
```

For help, run

```
python nanoparticlesize.py -h
```

## Sample output

## TODO

* Image processing / metadata processing to compute pixel-to-length scale
  from scale bar automatically
