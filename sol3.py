import numpy as np
import imageio as im
import scipy as si
import matplotlib.pyplot as plt
import os
from scipy.ndimage.interpolation import map_coordinates
from skimage.color import rgb2gray
GRAY_SCALE=2
RGB=3
MAX_PIXEL=255
##section 3.1
def build_gaussian_pyramid(im, max_levels, filter_size):
    row_filter,col_filter=build_filter(filter_size)
    pyr=[im]
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
    L_pyr.append(g_pyr[-1])
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
    :return: row_filter,col_filter
    """
    base=np.array([1,1])
    filter=base
    for j in range(1,filter_size-1):
        filter=np.convolve(base,filter)
    norm_fact=np.sum(filter)
    filter=filter*(1/norm_fact)
    return np.array(filter.reshape(1,(len(filter)))),np.array(filter.reshape((len(filter),1)))




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
##section 3.2
def laplacian_to_image(lpyr, filter_vec, coeff):
    lpyr[0]*=coeff[0]
    for i in range(len(lpyr)-1,0,-1):
        lpyr[i-1]=lpyr[i-1]+expand(coeff[i]*lpyr[i],filter_vec)
    return lpyr[0].astype('float64')

#section 3.3
def render_pyramid(pyr, levels):
    col=np.shape(pyr[0])[0]
    row = 0
    for level in range(levels):
        row+=np.shape(pyr[level])[1]
    return np.zeros((col,row))
def stretch(image, minimum, maximum):
    '''
    strech image to 0,1
    :param image:
    :param minimum: max value in image
    :param maximum: min value in image
    :return: streced image
    '''
    image =(image - minimum)/(maximum - minimum)
    image[image < 0] = 0
    image[image > 1] = 1
    return image
def display_pyramid(pyr, levels):
    res=render_pyramid(pyr, levels)
    ind=0
    for i in range(levels):
        N,M=np.shape(pyr[i])
        strech_im=stretch(pyr[i],np.min(pyr[i]),np.max(pyr[i]))
        res[0:N,ind:ind+M]=strech_im
        ind+=M
    plt.figure()
    plt.imshow(res, cmap='gray')
    plt.show()
##section 4
def pyramid_blending(im1,im2,mask,max_levels,filter_size_im,filter_size_mask):
    L1,filter_vec1=build_laplacian_pyramid(im1,max_levels,filter_size_im)
    L2,filter_vec2= build_laplacian_pyramid(im2, max_levels, filter_size_im)
    Gm,fil_m=build_gaussian_pyramid(mask.astype('float64'),max_levels,filter_size_mask)
    L_out=[]
    for k in range(max_levels):
        L_out_k=L1[k]*Gm[k]+L2[k]*(1-Gm[k])
        L_out.append(L_out_k)
    coeff=[1]*max_levels
    im_blend=laplacian_to_image(L_out,filter_vec1,coeff)
    return np.clip(im_blend,0,1)
#section 4.1
def blending_example1():
    im1=read_image(relpath('externals/lebron_res.jpg'),2)
    im2 = (im.imread(relpath('externals/tyron_res.jpg')) / 255).astype(
        'float64')
    mask=np.array((im.imread(relpath('externals/mask12.jpg')) / 255))
    mask=mask.astype('bool')
    im_blend=np.zeros((np.shape(im2)[0],np.shape(im2)[1],np.shape(im2)[2]))
    for i in range(3):
        im_blend[:,:,i]=pyramid_blending(im1[:,:,i],im2[:,:,i],mask,3,3,3)
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    ax4.imshow(im_blend)
    ax1.imshow(im1)
    ax2.imshow(im2)
    ax3.imshow(mask,cmap='gray')
    plt.show()
    plt.figure()
    plt.imshow(im_blend)
    plt.show()
    return im1,im2,mask,im_blend
def blending_example2():
    im1=read_image(relpath('externals/kramer.jpg'),2)
    im2 = (im.imread(relpath('externals/trump.jpg')) / 255).astype('float64')
    mask=np.array((im.imread(relpath('externals/mask2.jpg')) / 255))
    mask = mask.astype('bool')
    im_blend=np.zeros((np.shape(im2)[0],np.shape(im2)[1],np.shape(im2)[2]),
                      np.float64)
    for i in range(3):
        im_blend[:,:,i]=pyramid_blending(im1[:,:,i],im2[:,:,i],mask,3,3,3)
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    ax4.imshow(im_blend)
    ax1.imshow(im1)
    ax2.imshow(im2)
    ax3.imshow(mask,cmap='gray')
    plt.show()
    plt.figure()
    plt.imshow(im_blend)
    plt.show()
    return im1,im2,mask,im_blend
def read_image(filename, representation):
    #the function will read an image file and return a normalizes array of
    # its intesitys
    image=np.array(im.imread(filename).astype(np.float64))
    if np.amax(image)>1:
        image=np.array(image.astype(np.float64)/MAX_PIXEL)
    if representation==2 and image.ndim!=GRAY_SCALE:#return RGB from RGB file
        return image
    elif representation==1 and image.ndim==RGB:#return grayscale from RGB file
        return rgb2gray(image)
    elif representation==1 and image.ndim==GRAY_SCALE: #return grayscale from
        # grayscale file
        return image


def relpath(filename):
    return os.path.join(os.path.dirname(__file__), filename)





