import os, sys
import tensorflow as tf
from Model.pfld_model import PFLD
from Model.datasets import DateSet
from Model.utils import parse_arguments

# 恢复模型
# model = PFLD()
# checkpoint_dir = 'training_checkpoints'
# checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
# checkpoint = tf.train.Checkpoint(pfld=model)
# ckpt_manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=5)

# 如果存在检查点，恢复最新版本检查点
# if ckpt_manager.latest_checkpoint:
#   checkpoint.restore(ckpt_manager.latest_checkpoint)

# 加载模型
# model = tf.saved_model.load('models/pb/pfld_pb')
# model = tf.keras.models.load_model('models/h5/retinaface_train.h5')
model = tf.saved_model.load('models/pb/retinaface_pb')
# model = tf.keras.models.load_model('models/h5/pfld_infer.h5')
# model = tf.saved_model.load('models/pb/pfld_infer_pb')

# 保存为其它格式模型文件
# model.save('models/h5/pfld_1620.h5')
# tf.saved_model.save(model, 'models/pb/pfld_pb')
# tf.saved_model.save(model, 'models/pb/retinaface_pb')
# tf.saved_model.save(model, 'models/pb/pfld_infer_pb')
# tf.keras.utils.plot_model(model, to_file='pic/pfld_infer.png', show_shapes=True, dpi=96)


# 指定输入shape, 通过pb模型恢复
concrete_func = model.signatures[
  tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
concrete_func.inputs[0].set_shape([1, 640, 640, 3])


# 转换成tflite
# converter = tf.lite.TFLiteConverter.from_keras_model(model)
# converter = tf.lite.TFLiteConverter.from_saved_model('models/pb/pfld_infer_pb')
converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
# converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
# converter.target_spec.supported_types = [tf.lite.FLOAT16]
tflite_model = converter.convert()
# open("models/tflite/pfld_infer.tflite", "wb").write(tflite_model)
open("models/tflite/retinaface.tflite", "wb").write(tflite_model)
