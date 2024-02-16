import glob
import numpy as np
import rasterio
from rasterio.enums import Resampling



def load_coupled_imgs_labs(IMG_FOLDER, LAB_FOLDER, VERBOSE, IMG_SIZE, BANDS, MAX_IMG_VAL, MAX_LAB_VAL, CHANNELS_LAST, INVERT_BINARY_LABELS):

    images = glob.glob(IMG_FOLDER)
    labels = glob.glob(LAB_FOLDER)
    images.sort()
    labels.sort()
    
    fname = lambda full_path:full_path.split('/')[-1].split('.')[0] # we also remove file extension just in case images and labels are in different format

    image_list = []
    label_list = []
    
    for i in range(len(images)):
    
        if fname(images[i]) != fname(labels[i]):
            raise Exception("DataError: image file names and label file names do not match!")
        
        PRINT_STEP = 100
        if VERBOSE and (i+1)%PRINT_STEP==0: print('--Procesing image {}/{}...'.format(i+1, len(images)))
        
        OUT_SIZE = (BANDS,IMG_SIZE[0],IMG_SIZE[1])
        img = rasterio.open(images[i]).read(out_shape=OUT_SIZE, resampling=Resampling.cubic).astype('float32')/MAX_IMG_VAL # (bands, rows, cols)
        image_list.append(img)

        OUT_SIZE = (1,IMG_SIZE[0],IMG_SIZE[1])
        lab = rasterio.open(labels[i]).read(out_shape=OUT_SIZE, resampling=Resampling.nearest).astype('float32')/MAX_LAB_VAL # (1, rows, cols)
        if INVERT_BINARY_LABELS: # just in case we need to invert 0-1 labels to homogenize datasets
            lab = 1-lab
        label_list.append(np.rint(lab)) # rounding to [0,1] values (just for security)
        
    images = np.stack(image_list, axis=0) # (num_images, bands, rows, cols)
    labels = np.stack(label_list, axis=0) # (num_images, 1, rows, cols)

    if CHANNELS_LAST:
        images = np.moveaxis(images, 1, -1) # (num_images, rows, cols, bands)
        labels = np.moveaxis(labels, 1, -1) # (num_images, rows, cols, 1)

    return (images, labels)

  


def load_AISD(DATA_FOLDER, TST_SIZE=None, VAL_SIZE=None, RAND_STATE=0, VERBOSE=False, IMG_SIZE=(224,224), BANDS=3, MAX_IMG_VAL=255,  MAX_LAB_VAL=1, CHANNELS_LAST=False, INVERT_BINARY_LABELS=True):
    
    # Note that this dataset has the test and validation sets already created (however, we keep the 'TST_SIZE/VAL_SIZE' parameters for homogeneity)
    
    if VERBOSE: print('Loadidng AISD dataset...')
    
    TRA_IMG_FOLDER = DATA_FOLDER + "/Train412/shadow/*.tif"
    TRA_LAB_FOLDER = DATA_FOLDER + "/Train412/mask/*.tif"
    TST_IMG_FOLDER = DATA_FOLDER + "/Test51/shadow/*.tif"
    TST_LAB_FOLDER = DATA_FOLDER + "/Test51/mask/*.tif"
    VAL_IMG_FOLDER = DATA_FOLDER + "/Val51/shadow/*.tif"
    VAL_LAB_FOLDER = DATA_FOLDER + "/Val51/mask/*.tif"

    if VERBOSE: print('Training set...')
    tra_images, tra_labels = load_coupled_imgs_labs(TRA_IMG_FOLDER, TRA_LAB_FOLDER, VERBOSE, IMG_SIZE, BANDS, MAX_IMG_VAL, MAX_LAB_VAL, CHANNELS_LAST, INVERT_BINARY_LABELS)
    if VERBOSE: print('Test set...')
    tst_images, tst_labels = load_coupled_imgs_labs(TST_IMG_FOLDER, TST_LAB_FOLDER, VERBOSE, IMG_SIZE, BANDS, MAX_IMG_VAL, MAX_LAB_VAL, CHANNELS_LAST, INVERT_BINARY_LABELS)
    if VERBOSE: print('Validation set...')
    val_images, val_labels = load_coupled_imgs_labs(VAL_IMG_FOLDER, VAL_LAB_FOLDER, VERBOSE, IMG_SIZE, BANDS, MAX_IMG_VAL, MAX_LAB_VAL, CHANNELS_LAST, INVERT_BINARY_LABELS)
    
    return (tra_images, tra_labels, tst_images, tst_labels, val_images, val_labels)
    
    
