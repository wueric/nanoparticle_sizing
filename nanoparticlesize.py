import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from skimage.feature import peak_local_max
from skimage.segmentation import mark_boundaries
from skimage.morphology import watershed

from skimage import data, color
import skimage


from scipy import ndimage as ndi
from skimage.filters import sobel, threshold_otsu, threshold_isodata

from skimage import morphology

import skimage.filters as filters
from skimage.data import load
from skimage.color import rgb2gray

from matplotlib.backends.backend_pdf import PdfPages

import os
import argparse

from collections import namedtuple

plt.style.use('ggplot')
font = {'size' : 14}
matplotlib.rc('font', **font)

BOTTOM_BAR_HEIGHT = 38

ScaleDensityTuple = namedtuple('ScaleDensityTuple', ['scale', 'density'])

def segment_particles_high_density (image_path, pixel_size):
    image = load(image_path)
    image_gray = image[:-BOTTOM_BAR_HEIGHT,:]
    
    square_edges = sobel(image_gray)
    sobel_edges = square_edges / np.max(square_edges)
    despeckled = ndi.filters.median_filter(sobel_edges, size=(3,3))
    thresholded = despeckled < threshold_isodata(despeckled)
    
    distance = ndi.distance_transform_edt(thresholded)

    local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((21, 21)),
                                labels=thresholded)
    markers = ndi.label(local_maxi)[0]
    labels = watershed(-distance, markers, mask=thresholded)
    labels = morphology.remove_small_objects(labels, min_size=36)
    count = np.max(labels)
    
    
    # count areas
    particle_diameters = []
    for particle_index in range(count):
        num_pixels = (labels == particle_index).sum()
        
        area = num_pixels * pixel_size
        p_diameter = 2 * np.sqrt(area / np.pi)
        particle_diameters.append(p_diameter)        

    return count, labels, particle_diameters

def segment_particles_low_density (image_path, pixel_size):

    image = load(image_path)
    image_gray = image[:-BOTTOM_BAR_HEIGHT,:]

    thresholded = image_gray > threshold_otsu(image_gray)

    distance = ndi.distance_transform_edt(thresholded)
    local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((21, 21)),
                                labels=thresholded)
    markers = ndi.label(local_maxi)[0]
    labels = watershed(-distance, markers, mask=thresholded)
    count = np.max(labels)

    # count areas
    particle_diameters = []
    for particle_index in range(count):
        num_pixels = (labels == particle_index).sum()
        
        area = num_pixels * pixel_size
        p_diameter = 2 * np.sqrt(area / np.pi)
        particle_diameters.append(p_diameter)        

    return count, labels, particle_diameters


def generate_particle_segments_graphics (labels, name):

    pp = PdfPages(name)
    fig = plt.figure()

    super_marked = 1 - mark_boundaries(labels != 0, labels, color=(0,0,0))
    plt.imshow(super_marked[::-1,:])
    plt.xlim((0, labels.shape[1]))
    plt.ylim((0, labels.shape[0]))

    plt.tight_layout()

    pp.savefig()
    pp.close()

if __name__ == '__main__':

    parser = argparse.ArgumentParser("Size particles from SEM")
    parser.add_argument('mapfile',
                        type=str,
                        help='header: "histogram path, histogram title, histogram width, histogram height, histogram bin width in microns, histogram max bin"; standard line: "image_path,scale,HD_or_LD"')

    args = parser.parse_args()

    mapfile = args.mapfile

    paths_to_scale = {}
    working_directory = os.getcwd()


    histogram_path = "size_distribution"
    histogram_title = "Histogram"
    histogram_width = 8
    histogram_height = 4
    histogram_binwidth = 0.025
    histogram_binmax = 2.0

    histogram_particle_diameters = []

    with open(mapfile, 'r') as info:
        lines = info.readlines()

        iterator = lines.__iter__()

        first_line = iterator.next()
        histogram_parameters = first_line.split(',')

        try:
            histogram_path = str(histogram_parameters[0])
            histogram_title = str(histogram_parameters[1])
            histogram_width = int(histogram_parameters[2])
            histogram_height = int(histogram_parameters[3])
            histogram_binwidth = float(histogram_parameters[4])
            histogram_binmax = float(histogram_parameters[5])
        except:
            pass


        for line in iterator:
            line = line.strip('\n').split(',')
            
            current_path = line[0]
            sem_scale = float(line[1])
            
            particle_density = line[2]

            if not os.path.isabs(current_path):
                current_abs_path = os.path.abspath(current_path)
                paths_to_scale[current_abs_path] = ScaleDensityTuple(sem_scale, particle_density)
            else:
                paths_to_scale[current_path] = ScaleDensityTuple(sem_scale, particle_density)


    histogram_particle_diameters = []
    for path, datatuple in paths_to_scale.items():

        scale = datatuple.scale
        particle_density = datatuple.density

        pixel_area = scale ** 2


        num_particles, labeled_image, particle_diameters = None, None, None

        if particle_density == 'HD' or particle_density == 'hd':
            num_particles, labeled_image, particle_diameters = segment_particles_high_density(path, pixel_area)
        else:
            num_particles, labeled_image, particle_diameters = segment_particles_low_density(path, pixel_area)

        histogram_particle_diameters.extend(particle_diameters)


        sem_imagename = '.'.join(path.split('/')[-1].split('.')[:-1])
        segment_name = '{0}/segmented_{1}.pdf'.format(working_directory, sem_imagename)
        
        generate_particle_segments_graphics(labeled_image, segment_name)

    median_size = np.median(histogram_particle_diameters)
    mean_size = np.mean(histogram_particle_diameters)

    print '{0} um median size'.format(median_size)
    print '{0} um mean size'.format(mean_size)


    histogram_path = '{0}/{1}.pdf'.format(working_directory, histogram_path)
    pp = PdfPages(histogram_path)

    fig = plt.figure()

    fig.set_size_inches((histogram_width, histogram_height))

    bin_size = histogram_binwidth; min_edge = 0; max_edge = max(histogram_particle_diameters)
    #bin_size = 0.05; min_edge = 0; max_edge = HISTOGRAM_MAX_WIDTH
    N = (max_edge-min_edge)/bin_size; Nplus1 = N + 1
    bin_list = np.linspace(min_edge, max_edge, Nplus1)

    n, bins = np.histogram(histogram_particle_diameters, bins=bin_list)

    probability_mass_function = 1.0 * n / sum(n)
    plt.bar(bins[0:-1], probability_mass_function, width=bin_size)



    plt.title(histogram_title)
    plt.xlim(0, histogram_binmax)
    plt.xlabel("Diameter [um]")
    plt.ylabel("Probability")

    plt.tight_layout()

    pp.savefig()
    pp.close()
