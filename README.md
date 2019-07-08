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
### Example
```
a.show_image([im.image,im.im_pap,im.im_sipm],['Original','Paper','SiPM'])
```
![Three Images](/readme_images/three_images.png)

## Histograms:
```
# generate a histogram from an image
hist = a.histogram(image)
# display any number of histograms given in an array
a.show_hist([hist])
```
### Example
```
# generate histograms
hist = a.histogram(im.image)
hist_pap = a.histogram(im.im_pap)
hist_sipm = a.histogram(im.im_sipm)
# display histograms
a.show_hist([hist,hist_pap,hist_sipm],['Original','Paper','SiPM'])
```
![Three Histograms](/readme_images/three_histograms.png)

## Display Area Boundaries:
```
# generate an image with area boundaries drawn
edges = a.draw_boundaries(im)
# (watch out for aliasing, try displaying a bigger image if edges aren't clear)
```
### Example
```
# generate boundary image
edges = a.draw_boundaries(im,4)
# display image, I made this one bigger to avoid aliasing
plt.figure(figsize=(10,14))
plt.imshow(edges,cmap='gray')
plt.show()
```
[Boundaries](/readme_images/edges.png)

## Plotting Statistics:
```
# plot stats for an array of analyzed images
a.plot_stats([im0,im1,...])
```
### Example
Here I'm analyzing ten images with sequential naming. You must specify a pixel in both the cone and paper areas for each image.
```
fname_end = np.linspace(0,9,10, dtype=np.uint8)
im = np.full(10,cone.Image)
pap = np.array([500,500])
con = np.array([[1000,1200],
                [1200,1200],
                [1250,1250],
                [950,1200],
                [1300,1000],
                [1100,1100],
                [1150,1350],
                [1100,1150],
                [1350,1350],
                [1300,1300]])
for i in range(10):
    fname = 'filepath/IMG_299%s.JPG'%(str(fname_end[i]))
    im[i] = cone.Image(fname)
    a.analyze_image(im[i], pap, con[i])
a.plot_stats(im)
```
![Image Statistics](/readme_images/stats.png)
