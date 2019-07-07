# LightConeAnalysis

## Basic Usage:
```
import cone
a = cone.Analysis # define analysis functions
im = cone.Image(filename) # define image
paper = [x,y] # location of a pixel on the paper
cone = [x,y] # location of a pixel in the cone
stats = a.analyze_image(im, paper, cone) # analyze image and return statistics
im.image # the original image
im.im_pap # image of the paper
im.im_sipm # image of the SiPM
```

## Displaying Images:
```
# displays any number of images given in an array
a.show_image([image1,image2,...])
# use a.show_image([im.image]) to display the original image
```

## Histograms:
```
# generate a histogram from an image
hist = a.histogram(image)
# display any number of histograms given in an array
a.show_hist([hist])
```

# Display Area Boundaries:
```
# generate an image with area boundaries drawn
edges = a.draw_boundaries(im)
# (watch out for aliasing, try displaying a bigger image if edges aren't clear)
```

# Plotting Statistics:
```
# plot stats for an array of analyzed images
a.plot_stats([im0,im1,...])
```
