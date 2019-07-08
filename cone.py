##################################
# Light Cone Efficiency Analysis #
##################################
# Dana Zimmer
# Summer 2019

import matplotlib.pyplot as plt
import numpy as np
import cv2 

'''
READ ME:

Basic Usage:
import cone
a = cone.Analysis # define analysis functions
im = cone.Image(filename) # define image
paper = [x,y] # location of a pixel on the paper
cone = [x,y] # location of a pixel in the cone
stats = a.analyze_image(im, paper, cone) # analyze image and return statistics
im.image # the original image
im.im_pap # image of the paper
im.im_sipm # image of the SiPM

Displaying Images:
# displays any number of images given in an array
a.show_image([image1,image2,...])
# use a.show_image([im.image]) to display the original image

Histograms:
# generate a histogram from an image
hist = a.histogram(image)
# display any number of histograms given in an array
a.show_hist([hist])

Display Area Boundaries:
# generate an image with area boundaries drawn
edges = a.draw_boundaries(im)
# (watch out for aliasing, try displaying a bigger image if edges aren't clear)

Plotting Statistics:
# plot stats for an array of analyzed images
a.plot_stats([im0,im1,...])
'''

### IMAGE CLASS ###
# handles import and data of an image

class Image():

    def __init__(self, fname):
        # import file, 0 = grayscale
        self.fname = fname
        self.image = cv2.imread(fname, 0)
        # get basic attributes
        self.shape = self.image.shape
        self.npix = self.image.size



### ANALYSIS CLASS ###
# handles analysis functions

class Analysis():

    def __init__(self):
        return None

    ### MAIN ANALYSIS ROUTINE ###
    
    def analyze_image(self, im, pap, con):
        # generate binary image using Otsu's algorithm to find the best threshold value
        im.im_binary = self.threshold(im.image,0,typ=cv2.THRESH_OTSU)
        # get masks
        im.mask_pap = self.floodfill_mask(im.im_binary, pap)
        im.mask_con = self.floodfill_mask(im.im_binary, con)
        im.mask_sipm = self.get_sipm_mask(im)
        # separate parts of image
        im.im_pap = self.separate_image(im.image, im.mask_pap)
        im.im_con = self.separate_image(im.image, im.mask_con)
        im.im_sipm = self.separate_image(im.image, im.mask_sipm)
        # get statistics
        im.stats = self.get_stats(im.im_pap, im.im_sipm)
        return im.stats



    ### ANALYSIS METHODS ###
    
    # basic threshold, can choose different types, see opencv docs
    def threshold(self, image, thresh, maxval=255, typ=None):
        if typ is not None:
            ret, imthres = cv2.threshold(image, thresh, maxval, typ)
        else:
            ret, imthres = cv2.threshold(image, thresh, maxval)
        return imthres

    # calculates a histogram from an image
    def histogram(self, image, mask=None):
        hist = cv2.calcHist([image],channels=[0],mask=mask,histSize=[256],ranges=[0,256])
        return hist

    # creates an 8-bit blank canvas of size (x,y) and value val
    def blank(self, size, val):
        if val is 'black': canvas = np.zeros(size, dtype=np.uint8)
        elif val is 'white': canvas = cv2.bitwise_not(np.zeros(size, dtype=np.uint8))
        elif type(val) is int: canvas = np.full(size, val, dtype=np.uint8)
        else: 
            print('Error: Value type %s not understood in blank().'%(str(type(val))))
            return None
        return canvas

    # creates a mask of the area floodfilled starting at location loc
    def floodfill_mask(self, image, loc):
        # initialize floodfill parameters
        x,y = loc[:2] # location of starting pixel
        floodfill = image.copy() # copy of image, floodfill will rewrite
        mask = self.blank((image.shape[0]+2,image.shape[1]+2),'black') # blank black mask, floodfill requires extra 1 pixel border
        # perform floodfill, generating a mask of filled pixels
        cv2.floodFill(floodfill, mask, (x,y), 0)
        # change true/false values in mask to 255/0 instead of 1/0
        rows,cols = np.where(mask == 1)
        mask[rows,cols] = 255
        # remove extra pixel border
        mask = mask[1:-1,1:-1]
        return mask

    # separates the part of the original image given by a mask
    def separate_image(self, image, mask):
        white = self.blank(image.shape, 'white') # blank white image
        part_image = cv2.bitwise_and(image, white, mask=mask) # get partial image
        # replace 0s with NaNs
        rows,cols = np.where(part_image==0) # find 0 indices
        part_image_nan = np.array(part_image, dtype=np.float32) # convert to float (since NaN is a float
        part_image_nan[rows,cols] = np.nan # replace with NaN
        return part_image_nan

    # generates a mask for the SiPM area based on the size of the mount opening mask
    def get_sipm_mask(self, im):
        rows,cols = np.where(im.mask_con!=0) # locations of pixels in the cone
        radius = np.sqrt(rows.size/np.pi) # radius in pixels
        pixpermm = 2*radius/11 # 11mm outer diameter
        xlow,xhigh = cols[0],cols[-1]
        ylow,yhigh = rows[0],rows[-1]
        center = (int((xhigh-xlow)/2.0+xlow), int((yhigh-ylow)/2.0+ylow)) # center pixel
        halfsidelen = int(pixpermm*3) # 6mm SiPM size
        upper_corner = (center[0]-halfsidelen, center[1]-halfsidelen)
        lower_corner = (center[0]+halfsidelen, center[1]+halfsidelen)
        mask_sipm = self.blank(im.mask_con.shape, 'black')
        cv2.rectangle(mask_sipm, upper_corner, lower_corner, 255, thickness=-1)
        im.center = center
        im.pixpermm = pixpermm
        im.sipm_corners = (upper_corner,lower_corner)
        return mask_sipm

    # calculate statistics for an image
    def get_stats(self, im_pap, im_sipm):
        stats = {'mean_pap' : np.nanmean(im_pap),
                 'mean_sipm' : np.nanmean(im_sipm),
                 'std_pap' : np.nanstd(im_pap),
                 'std_sipm' : np.nanstd(im_sipm),
                 'ratio' : np.nanmean(im_sipm)/np.nanmean(im_pap)}
        return stats

    # draws lines 2*npix wide around image boundaries
    def draw_boundaries(self, im, npix=5):
        image = im.image.copy()
        outer_radius = im.pixpermm*11/2 # 11mm opening circle in mount
        inner_radius = im.pixpermm*np.sqrt(2*3**2) # circle around SiPM
        cv2.circle(image, im.center, int(outer_radius), 255, thickness=npix*2) # draw circle around mount opening, outer radius
        cv2.circle(image, im.center, int(inner_radius), 255, thickness=npix*2) # draw circle around SiPM, inner radius
        cv2.rectangle(image, im.sipm_corners[0], im.sipm_corners[1], 255, thickness=npix*2) # draw rectangle around SiPM
        paper_edges = self.get_edges(im.mask_pap, npix)  # get edges of paper boundary
        rows,cols = np.where(paper_edges==255) # locations of edges
        image[rows,cols] = 255 # draw paper boundary
        return image

    # returns mask of edges (of a binary mask) 2*npix wide
    def get_edges(self, mask, npix):
        edges = self.blank(mask.shape, 'black') # blank image
        edges = cv2.Canny(mask,0,256) # Canny edge detection
        rows,cols = np.where(edges==255) # locations of edges
        edges_border = self.blank((mask.shape[0]+npix*2, mask.shape[1]+npix*2), 'black') # add a n-pixel border
        for n in range(npix): # fill in n pixels around edges
            edges_border[rows+n,cols] = 255
            edges_border[rows-n,cols] = 255
            edges_border[rows,cols+n] = 255
            edges_border[rows,cols-n] = 255
        edges = edges_border[npix:-npix,npix:-npix] # copy into array with no border
        return edges

    ### DISPLAY METHODS ###
    
    # display any number of images, given in an array
    def show_image(self, im, tit=None):
        n = len(im) # number of images in array
        plt.figure(figsize=(5*n, 5))
        for i in range(n):
            plt.subplot(1, n, i+1)
            plt.imshow(im[i], cmap='gray')
            plt.xticks([]), plt.yticks([])
            plt.colorbar(pad=0), plt.clim(0,256)
            if tit is not None: plt.title(tit[i])
        plt.show()

    # displays any number of histograms, given in an array
    def show_hist(self, hist, tit=None, xlim=None, ylim=None):
        n = len(hist)
        plt.figure(figsize=(5*n, 3))
        for i in range(n):
            plt.subplot(1, n, i+1)
            plt.bar(range(0,256), hist[i])
            if xlim is not None: plt.xlim(xlim)
            if ylim is not None: plt.ylim(ylim)
            plt.xlabel('Pixel Value'), plt.ylabel('Number of Pixels')
            if tit is not None: plt.title(tit[i])
        plt.show()

    # plots statistics for an array of images, must run analyze_image() first
    def plot_stats(self, im):
        n = len(im)
        x = np.linspace(0,n-1,n)
        mean_pap, mean_sipm = np.zeros(n), np.zeros(n)        
        std_pap, std_sipm = np.zeros(n), np.zeros(n)
        ratio = np.zeros(n)
        for i in range(n):
            mean_pap[i] = im[i].stats['mean_pap']
            mean_sipm[i] = im[i].stats['mean_sipm']
            std_pap[i] = im[i].stats['std_pap']
            std_sipm[i] = im[i].stats['std_sipm']
            ratio[i] = im[i].stats['ratio']        
        plt.figure()
        ax1 = plt.subplot(111)
        ax2 = ax1.twinx()
        ax1.set_ylim(0,265)
        ax2.set_ylim(0,max(ratio)+1)
        ax1.set_xlabel('Picture')
        ax1.set_ylabel('Pixel Value [8-bit]')
        ax2.set_ylabel('Ratio SiPM/Paper')
        line1 = ax1.errorbar(x, mean_pap, yerr=std_pap, linestyle='--', marker='x', label='Mean Paper Pixel Value')
        line2 = ax1.errorbar(x, mean_sipm, yerr=std_sipm, linestyle='--', marker='x', label='Mean SiPM Pixel Value')
        line3 = ax2.plot(x, ratio, marker='x', color='g', label='Ratio')
        ax1.legend(loc='upper right')
        ax2.legend(loc='upper left')
        plt.show()
