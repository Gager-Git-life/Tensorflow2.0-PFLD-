import cv2
import time
import os, sys
import numpy as np
import tensorflow as tf
from Model.datasets import DateSet
from Model.utils import parse_arguments, Normalization, Draw_lamdmark

args = parse_arguments(sys.argv[1:])
test_dataset, num_test_file = DateSet(args.test_list, args, False)
batch_test_dataset = test_dataset.batch(60)
test_iterator = iter(batch_test_dataset)
test_data  = next(test_iterator)
test_images = test_data[0]

model = tf.keras.models.load_model('models/h5/pfld_infer.h5', compile=False)
print(model.summary())
tf.keras.utils.plot_model(model, to_file='pic/pfld_infer.png', show_shapes=True, dpi=96)

bt = time.time()
landmarks = model(test_images, training=False)
print("[INFO]>>> ust time:{}".format((time.time()-bt)))

# landmarks = Normalization(landmarks)
# landmarks = np.array(landmarks*112)
# test_images = np.array(test_images)
# print('[INFO]>>> test image len:{}'.format(len(test_images)))
# out_img = Draw_lamdmark(test_images[0], landmarks[0])
# out_img = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
# cv2.imwrite('pic/out_.jpg', out_img*255)





