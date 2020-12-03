import numpy as np
import imageio as im
import scipy as si
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import map_coordinates
from skimage.color import rgb2gray
GRAY_SCALE=2
RGB=3
MAX_PIXEL=255
##section 1
def build_gaussian_pyramid(im, max_levels, filter_size):
    row_filter,col_filter=build_filter(filter_size)
    pyr=[list(im)]
    cur=im
    for level in range(max_levels-1):
        G_i=filter_(cur,row_filter,0)
        G_i = filter_(G_i, col_filter,0)
        G_i=G_i[::2,::2]#sample
        if np.shape(G_i)[0]<16 or np.shape(G_i)[1]<16:
            break
        pyr.append(G_i)
        cur=G_i
    return pyr,row_filter

def build_laplacian_pyramid(im, max_levels, filter_size):
    g_pyr,filter_vect=build_gaussian_pyramid(im, max_levels, filter_size)
    L_pyr=[]
    for level in range(len(g_pyr)-1):
        L_i=g_pyr[level]-expand(g_pyr[level+1],filter_vect)
        L_pyr.append(L_i)
    L_pyr.append(L_pyr[-1])
    return L_pyr,filter_vect
def expand(im,filter_vect):
    m, n = im.shape
    out = np.zeros((2*m, 2 * n), dtype=im.dtype)
    out[::2,::2] = im
    out=filter_(out,filter_vect,1)
    out = filter_(out, filter_vect.reshape((len(filter_vect[0]),1)), 1)
    return out

def build_filter(filter_size):
    """
    generate binomial filter vector
    :param size:
    :param row_or_col:
    :return: row_filter,col_filter
    """
    base=np.array([1,1])
    filter=base
    for j in range(1,filter_size-1):
        filter=np.convolve(base,filter)
    norm_fact=np.sum(filter)
    filter=filter*(1/norm_fact)
    return np.array(filter.reshape(1,(len(filter)))),np.array(filter.reshape((
        len(filter),1)))



def filter_(im,filter,expand_or_reduce):
    """
    apply the convolution with the filter
    :param im:
    :param filter:
    :param expand_or_reduce:
    :return: filtered image
    """
    if expand_or_reduce==0:#case of reduction
        filtered_im=si.ndimage.filters.convolve(im,filter)
        #filtered_im=si.signal.convolve2d(im,filter, 'same')
    else:#case of expand
        filter = 2* filter
        filtered_im = si.ndimage.filters.convolve(im,filter)
        #filtered_im = si.signal.convolve2d(im, filter, 'same')
    return filtered_im


def read_image(filename, representation):
    #the function will read an image file and return a normalizes array of
    # its intesitys
    image=im.imread(filename).astype(np.float64)
    if np.amax(image)>1:
        image=image.astype(np.float64)/MAX_PIXEL
    if representation==2 and image.ndim!=GRAY_SCALE:#return RGB from RGB file
        return image
    elif representation==1 and image.ndim==RGB:#return grayscale from RGB file
        return rgb2gray(image)
    elif representation==1 and image.ndim==GRAY_SCALE: #return grayscale from
        # grayscale file
        return image
im=read_image('monkey.jpg',1)
#im=np.array([[1,3,1,3],[1,3,1,3],[1,3,1,3],[1,3,1,3]])
pyr,filter=build_laplacian_pyramid(im, 5,5)
pyr_g,filter1=build_gaussian_pyramid(im, 5,5)
#print(len(pyr[0]),len(pyr[1]),len(pyr[3]),len(pyr))
plt.figure()
plt.imshow(pyr[4],cmap='gray')
plt.show()
plt.figure()
plt.imshow(pyr[4],cmap='gray')
plt.show()
#filter_1,s=build_filter(3)
#print(np.mean(expand(pyr[1],filter)))
#print(np.mean(im))
print(pyr[4])
#print(pyr_g[4])