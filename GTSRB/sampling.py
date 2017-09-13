import glob
import os
import cv2
from sklearn.utils import resample
import pickle
import numpy as np
features = []
labels = []
for label in range(0, 43):
    files = glob.glob(os.path.join(os.path.dirname(__file__), 'Images', '{0:05d}'.format(label), '*.ppm'),
                      recursive=True)
    samples = resample(files, n_samples=1)
    for sample in samples:
        img = cv2.imread(sample)
        img = cv2.resize(img, (32, 32))
        features.append(img)
        labels.append(label)
        cv2.imwrite(os.path.join(os.path.dirname(__file__), '..', 'my-data', '{0:05d}.png').format(label),img)

pickle.dump({
    'features': np.array(features),
    'labels': np.array(labels)
}, open(os.path.join(os.path.dirname(__file__), '..', 'my-data', 'my-data.p'), 'wb'))
