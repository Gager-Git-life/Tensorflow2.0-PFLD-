import time
import datetime
import os, sys
import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers
from Model.datasets import DateSet
from Model.pfld_model import PFLD, AuxiliaryNet
from Model.loss import pfld_loss
from Model.utils import parse_arguments, generate_and_save_images, set_memory_growth, ProgressBar, prepare
from Model.lr_scheduler import MultiStepWarmUpLR

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# set_memory_growth()
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

args = parse_arguments(sys.argv[1:])

init_lr = args.learning_rate

# 准备训练和测试数据集
train_dataset, num_train_file = DateSet(args.file_list, args, False)
test_dataset, num_test_file = DateSet(args.test_list, args, False)

batch_train_dataset = train_dataset.repeat(args.max_epoch).shuffle(1000)
batch_train_dataset = batch_train_dataset.batch(args.batch_size)
train_iterator = iter(batch_train_dataset)

# batch_test_dataset = test_dataset.batch(args.batch_size).repeat()
if(args.first_train):
    prepare(args)
    batch_test_dataset = test_dataset.batch(4)
    test_iterator = iter(batch_test_dataset)

    test_data  = next(test_iterator)
    test_images = test_data[0]
    test_landmarks = test_data[1]

    np.save('images.npy', test_images)
    np.save('landmarks.npy', test_landmarks)
else:
    test_images = np.load('images.npy')
    test_landmarks = np.load('landmarks.npy')


# 模型准备
pfld_model = PFLD()
auxiliary_model = AuxiliaryNet()

pfld_model.summary()
auxiliary_model.summary()
tf.keras.utils.plot_model(pfld_model, to_file='pic/pfld.png', show_shapes=True, dpi=96)
tf.keras.utils.plot_model(auxiliary_model, to_file='pic/auxiliary.png', show_shapes=True, dpi=96)


# 优化器选择
steps_per_epoch = num_train_file // args.batch_size
# values     = [1.4e-4,   1.4e-4,    1.4e-4,   1.4e-4]
# boundaries = [0-1,    1-3,     3-6,      6-40]
# warmup_steps = 1
learning_rate = MultiStepWarmUpLR(
    initial_learning_rate=init_lr,
    lr_steps=[e * steps_per_epoch for e in [3, 6, 40]],
    lr_rate=[1, 1.0, 0.8], #[1e-4, 1.4e-4, 1.5e-4, 1e-6]
    warmup_steps=steps_per_epoch,
    min_lr=1e-7)

pfld_optimizer = optimizers.Adam(learning_rate)
auxi_optimizer = optimizers.Adam(1e-4)


# 模型存储
checkpoint_dir = 'training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(step=tf.Variable(0, name='step'),
                                 pfld_optimizer=pfld_optimizer,
                                 pfld=pfld_model,
                                 auxiliary_optimizer=auxi_optimizer,
                                 auxiliary=auxiliary_model
                                 )

ckpt_manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=5)

# 如果存在检查点，恢复最新版本检查点
if ckpt_manager.latest_checkpoint:
    checkpoint.restore(ckpt_manager.latest_checkpoint)
    print('[INFO]>>> load ckpt from {} at step {}.'.format(
        ckpt_manager.latest_checkpoint, checkpoint.step.numpy()))

per_steps = checkpoint.step.numpy()
if(per_steps > 0):
    print('[INFO]>>> find per train steps:{}'.format(per_steps))

# 训练进度条
prog_bar = ProgressBar(steps_per_epoch,
                       checkpoint.step.numpy() % steps_per_epoch)

# 训练日志设置
log_dir = "logs/"
summary_writer = tf.summary.create_file_writer(
  log_dir + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

@tf.function
def train_step(pfld, auxiliary, dataloader):
    img, landmark_gt, attribute_gt, euler_angle_gt = dataloader
    with tf.GradientTape() as tap_pfld, tf.GradientTape() as tap_auxi:
        featu,landmarks_pre = pfld(img, training=True)
        angle_pre = auxiliary(featu, training=True)
        # 计算损失
        sum_loss, l2_loss, angle_loss = pfld_loss(attribute_gt, landmark_gt, euler_angle_gt,
                                        angle_pre, landmarks_pre, args.batch_size)

    pfld_gradients = tap_pfld.gradient(l2_loss, pfld.trainable_variables)
    auxi_gradients = tap_auxi.gradient(angle_loss, auxiliary.trainable_variables)
    pfld_optimizer.apply_gradients(zip(pfld_gradients, pfld.trainable_variables))
    auxi_optimizer.apply_gradients(zip(auxi_gradients, auxiliary.trainable_variables))

    return sum_loss, l2_loss, angle_loss

# 训练
def train(pfld_model, auxiliary_model):
    for epoch in range(args.max_epoch):
        print("[INFO]>>> epoch:{}".format(epoch))
        for i, img_batch in enumerate(train_iterator):
            checkpoint.step.assign_add(1)
            steps = checkpoint.step.numpy()
            # step_start = time.time()
            sum_loss, l2_loss, angle_loss = train_step(pfld_model, auxiliary_model, img_batch)
            prog_bar.update("epoch={}/{}, step{}, sum_loss={:.4f}, l2_loss:{:.4f}, angle_loss:{:.4f}, lr={:.1e}".format(
                ((steps - 1) // steps_per_epoch) + 1, args.max_epoch, steps, sum_loss, l2_loss,
                angle_loss, pfld_optimizer.lr(steps).numpy()))
            # logs = 'Epoch={}, step{}, sum_loss:{},l2_loss:{}, angle_loss:{} use_time:{}'
            # tf.print(tf.strings.format(logs, (epoch, steps, sum_loss, l2_loss, angle_loss, time.time()-step_start)), output_stream=sys.stdout)

            if(steps % 2 == 0):
                with summary_writer.as_default():
                    tf.summary.scalar('sum_loss', sum_loss, step=steps)
                    tf.summary.scalar('l2_loss', l2_loss, step=steps)
                    tf.summary.scalar('angle_loss', angle_loss, step=steps)
                    tf.summary.scalar('pfld_learning_rate', pfld_optimizer.lr(steps), step=steps)

            if(steps % 50 == 0):
                ctime = time.strftime("%Y/%m/%d/ %H:%M:%S")
                ckpt_save_path = ckpt_manager.save()
                logs = 'Time:{}, Epoch={}, save_path={}'
                tf.print(tf.strings.format(logs, (ctime, epoch, ckpt_save_path)), output_stream=sys.stdout)
                generate_and_save_images(pfld_model, int(steps/50), test_images, test_landmarks)


if __name__ == "__main__":
    train(pfld_model, auxiliary_model)
    pfld_model.save('models/pfld_new.h5')
    auxiliary_model.save('models/auxiliary_new.h5')

