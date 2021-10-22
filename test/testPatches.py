import numpy as np 
import scipy as sp
from numpy import linalg as LA
from sklearn import preprocessing
from skimage import io, util, data
from skimage.io import imread, imshow
from skimage.color import rgb2gray
from sklearn.feature_extraction import image
from skimage.transform import rescale, resize, downscale_local_mean
import requests
import json
from skimage import exposure
from scipy import ndimage
import matplotlib.pyplot as plt
from sklearn.datasets import load_sample_images
import time
'''
images = all test images from sklearn
imgDims = dimensions of all test images
imgIndex = the chosen image for the test, 0 - (len(images)-1)
atoms = number of atoms in the dictionary
numberPods = number of pods in Kubernetes during test
patch_size = dimensions of each patch, m = patch_width x patch_height
'''
 
#imshow(images[5], cmap='gray')
#Load 10 real images gray-scaled
dataset = load_sample_images()   
images = []
images.append(io.imread('castle.jpg', as_gray=True).astype(float))
images.append(io.imread('lenna.jpg', as_gray=True).astype(float))
images.append(rgb2gray(dataset.images[0]))
images.append(rgb2gray(dataset.images[1]))
images.append(rgb2gray(data.chelsea()))
images.append(rgb2gray(data.camera()))
images.append(rgb2gray(data.coffee()))
images.append(rgb2gray(data.rocket()))
images.append(rgb2gray(data.astronaut()))
images.append(rgb2gray(data.astronaut()[30:180, 150:300]))

'''
img = images[2]
imshow(img)
#gamma_corrected = exposure.adjust_gamma(img, 2)
#logarithmic_corrected = exposure.adjust_log(img, 1)

hist, bin_edges = np.histogram(img, bins=60)
bin_centers = 0.5*(bin_edges[:-1] + bin_edges[1:])
binary_img = img > 0.5

plt.figure(figsize=(11,4))
plt.subplot(131)
plt.imshow(img)
plt.axis('off')
plt.subplot(132)
plt.plot(bin_centers, hist, lw=2)
plt.axvline(0.5, color='r', ls='--', lw=2)
plt.text(0.57, 0.8, 'histogram', fontsize=20, transform = plt.gca().transAxes)
plt.yticks([])
plt.subplot(133)
plt.imshow(binary_img, cmap=plt.cm.gray, interpolation='nearest')
plt.axis('off')
plt.subplots_adjust(wspace=0.02, hspace=0.3, top=1, bottom=0.1, left=0, right=1)
plt.show()
'''

#Downsample images by a factor along each axis.
for i, s in enumerate(images):
   images[i] = downscale_local_mean(images[i], (2, 2))
    

   
#Save dimensions
imgDims = []
for img in images:
    height, width = img.shape
    imgDims.append((height, width))
    
#Setup for patches
N = 100
numberPods = 2
patchSize = (5, 5)
imgIndex = 9
imshow(images[imgIndex])

allPatches = image.extract_patches_2d(images[imgIndex][:, :], patchSize)

patchDims = allPatches.shape

allPatches_reshaped = allPatches.reshape(allPatches.shape[0], -1)

#patchMean = np.mean(allPatches_reshaped, axis=0)
#allPatches_reshaped -= patchMean

#patchStd = np.std(allPatches_reshaped, axis=0)
#allPatches_reshaped /= patchStd

allPatches_reshaped = np.transpose(allPatches_reshaped)


split_size = int(np.floor((patchDims[0]/numberPods)))
imgPatches = []

for i in range(0,numberPods):
    imgPatches.append(allPatches_reshaped[:, range(split_size*i, split_size*(i+1))])


'''
#Extract reference patches for as many pods we have for some image
patchDims = allPatches.shape
split_size = int(np.floor((patchDims[0]/numberPods)))
imgPatches = []

for i in range(0,numberPods):
    imgPatches.append(allPatches[range(split_size*i, split_size*(i+1)), :, : ])
    imgPatches[i] = imgPatches[i].reshape(imgPatches[i].shape[0], -1)
    imgPatches[i] -= np.mean(imgPatches[i], axis=0) # normalization
    imgPatches[i] /= np.std(imgPatches[i], axis=0) # standardizing
    imgPatches[i] = np.transpose(imgPatches[i]) # transpose
'''
#Make D with random data
D = np.matrix(np.random.rand(np.shape(imgPatches[0])[0],N))



# %% Send data
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


#Make HTTP post
url = 'http://192.168.1.111:30470/load_data/'

payload_D = D.tolist()
payload_Y = json.dumps(imgPatches, cls=NumpyEncoder)

payload = {'D': payload_D,'Y': payload_Y}
response = requests.post(url,json=payload)
print(response)

# %% Start work

url = 'http://192.168.1.111:30470/start_work/'
response = requests.post(url)
print(response)

# %% Get training data Respons

response = requests.get('http://192.168.1.111:30654/training_data/')

print(response.content)

# %% Get Results

response = requests.get('http://192.168.1.111:31664/get_results/')

print(response)

D_new = []
X_new = []
stats = []

for res in response.json():
    D_new.append(res['D'])
    X_new.append(res['X'])
    stats.append(res['S'])


# %% Convert and save data
    
Y_new = np.dot(np.array(D_new[0]),np.array(X_new[0]))

for i in range(1, len(D_new)):
    Y_new = np.concatenate((Y_new, np.dot(np.array(D_new[i]),np.array(X_new[i]))), axis=1)

#Y_new1 = np.dot(np.array(D_new[0]),np.array(X_new[0]))
#Y_new2 = np.dot(np.array(D_new[1]),np.array(X_new[1]))
#Y_new = np.concatenate((Y_new1, Y_new2), axis=1)

Y_new_trans = np.transpose(Y_new)


#Y_new_trans *= patchStd
#Y_new_trans += patchMean



patches_new = Y_new_trans.reshape(np.size(Y_new_trans,0), *patchSize)

reconstruction = image.reconstruct_from_patches_2d(patches_new, (imgDims[imgIndex]))

# %% Save data
timestr = time.strftime("%Y%m%d-%H%M%S")
f = open(timestr + ".txt","w+")
f.write(str(stats))
f.write(str(D_new))
f.write(str(X_new))
f.write(str(Y_new_trans))
f.write(str(reconstruction))
f.close()


# %% Show data
def show_with_diff(image, reference, title):
    """Helper function to display denoising"""
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.title('Reconstruction')
    plt.imshow(image, vmin=0, vmax=1, cmap=plt.cm.gray,
               interpolation='nearest')
    plt.xticks(())
    plt.yticks(())
    plt.subplot(1, 3, 2)
    difference = image - reference

    plt.title('Difference (norm: %.2f)' % np.sqrt(np.sum(difference ** 2)))
    plt.imshow(difference, vmin=-0.5, vmax=0.5, cmap=plt.cm.PuOr,
               interpolation='nearest')
    plt.xticks(())
    plt.yticks(())
    plt.subplot(1, 3, 3)
    plt.title('Original')
    plt.imshow(reference, vmin=0, vmax=1, cmap=plt.cm.gray, 
               interpolation='nearest')
    plt.xticks(())
    plt.yticks(())
    plt.suptitle(title, size=16)
    plt.subplots_adjust(0.02, 0.02, 0.98, 0.79, 0.02, 0.2)
    plt.savefig('image_after_reconstruction.pdf')


show_with_diff(reconstruction, images[imgIndex], 'Distorted image')    

#plt.imshow(reconstruction, vmin=0, vmax=1, cmap=plt.cm.gray, interpolation='nearest')

#plt.imshow(images[imgIndex], vmin=0, vmax=1, cmap=plt.cm.gray, interpolation='nearest')


# %%
'''
#do reconstruction of the image that was used as patches, see split_portion
Y = np.dot(np.array(D),np.array(X))
Y += np.mean(data, axis=0)
Y = Y.reshape(len(data), *patch_size)
result = image.reconstruct_from_patches_2d(Y, (height, width // split_portion))

#difference between org and reconstructed
difference = result - reference[:, :width // split_portion ]
print('Difference for result (norm: %.2f)' % np.sqrt(np.sum(difference ** 2)))
'''
'''
# Show the patches in the dictionary-
plt.figure(figsize=(4.2, 4))
for i, comp in enumerate(V[:100]):
    plt.subplot(10, 10, i + 1)
    plt.imshow(comp.reshape(patch_size), cmap=plt.cm.gray_r,
               interpolation='nearest')
    plt.xticks(())
    plt.yticks(())
plt.suptitle('Dictionary learned from patches\n' +
             'Train time %.1fs on %d patches' % (dt, len(data)),
             fontsize=16)
plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)

#Random signal and dictionary
amount = 10  #Samples of atoms for each class; K = len(classes)*amount
signals = 20 #Samples in Y
atoms = 15
TestDictionary = np.matrix(np.random.rand(10,15)) #random matricies for debugging
TestSignal = np.matrix(np.random.rand(10,20))
S = signals			#Same as in paper
K = np.shape(TestDictionary)[1]  #Same as in paper 
ddim = np.shape(TestDictionary)[0]

tD = 5 #cloud KSVD iterations
t0 = 5 #sparsity
tc = 2 #consensus iterations
tp = 2 #power iterations

print(tD, t0, tc, tp)

nodes = 1
weights = np.zeros(nodes) #the weights for nodes, number if nodes is 1
refvec = np.matrix(np.ones((ddim,1))) #Q_init for power method, sets direction of result
Tag = 11			  #Transmission tag, ensures MPI transmissions don't interfere
CorrectiveSpacing = 3 #Regular iterations before a corrective iteration
timeOut = 0.150		  #Time nodes wait before they move on

print('Starting C-KSVD')
rt0 = time.time()
D,X,rerror = CloudKSVD(TestDictionary,TestSignal,refvec,tD,t0,tc,tp,weights,Tag,CorrectiveSpacing,timeOut)
rt1 = time.time()
rt = rt1 - rt0
errorCloudKsvd = np.linalg.norm(TestSignal-np.dot(D,X))**2 #L2 norm squared for error
randDic = np.matrix(np.random.rand(10,15))
errorRandom = np.linalg.norm(TestSignal-np.dot(randDic,X))**2 #L2 norm squared for error
print('Time to run Cloud K-SVD: %.2f' % (rt))
print('Error with random dictionary: %.2f' % (errorRandom))
print('Error with CK-SVD dictionary: %.2f' % (errorCloudKsvd))
'''