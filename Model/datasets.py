import tensorflow as tf
import numpy as np

def DateSet(file_list, args, debug=False):
    file_list, landmarks, attributes,euler_angles = gen_data(file_list)
    if debug:
        n = args.batch_size * 10
        file_list = file_list[:n]
        landmarks = landmarks[:n]
        attributes = attributes[:n]
        euler_angles=euler_angles[:n]
    dataset = tf.data.Dataset.from_tensor_slices((file_list, landmarks, attributes,euler_angles))

    def _parse_data(filename, landmarks, attributes,euler_angles):
        # filename, landmarks, attributes = data
        file_contents = tf.io.read_file(filename)
        image = tf.image.decode_png(file_contents, channels=args.image_channels)
        # print(image.get_shape())
        # image.set_shape((args.image_size, args.image_size, args.image_channels))
        # image = tf.image.resize(image, (args.image_size, args.image_size))
        image = tf.cast(image, tf.float32)

        image = (image / 127.5 - 1.0)
        return (image, landmarks, attributes, euler_angles)

    dataset = dataset.map(_parse_data)
    dataset = dataset.shuffle(buffer_size=10000)
    return dataset, len(file_list)

def gen_data(file_list):
    with open(file_list,'r') as f:
        lines = f.readlines()
    filenames, landmarks,attributes,euler_angles = [], [], [],[]
    for line in lines:
        line = line.strip().split()
        path = line[0]
        landmark = line[1:197]
        # landmark = landmark_slice(landmark, count=1)
        attribute = line[197:203]
        euler_angle = line[203:206]

        landmark = np.asarray(landmark, dtype=np.float32)
        attribute = np.asarray(attribute, dtype=np.int32)
        euler_angle = np.asarray(euler_angle,dtype=np.float32)
        filenames.append(path)
        landmarks.append(landmark)
        attributes.append(attribute)
        euler_angles.append(euler_angle)
        
    filenames = np.asarray(filenames, dtype=np.str)
    landmarks = np.asarray(landmarks, dtype=np.float32)
    attributes = np.asarray(attributes, dtype=np.int32)
    euler_angles = np.asarray(euler_angles,dtype=np.float32)
    return (filenames, landmarks, attributes,euler_angles)


def landmark_slice(landmarks, count):
    # print(len(landmarks))
    x = landmarks[0:-1:count*4]
    y = landmarks[1:-1:count*4]

    out_landmarks = []
    for x_, y_ in zip(x,y):
        out_landmarks.extend([x_, y_])

    # print(len(out_landmarks))
    return out_landmarks
