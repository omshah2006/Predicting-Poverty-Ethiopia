from batcher import Batcher

INPUT_SHAPE = (224, 224, 3)
BUCKET = False

TFRECORDS_DIR = '../data/lsms_tfrecords/'

# Split dataset
train_batcher = Batcher(
    image_shape=INPUT_SHAPE, bucket=BUCKET, batch_size=4559, shuffle=False, repeat=1, split="train"
).get_dataset()

val_batcher = Batcher(
    image_shape=INPUT_SHAPE, bucket=BUCKET, batch_size=1302, shuffle=False, repeat=1, split="val"
).get_dataset()

test_batcher = Batcher(
    image_shape=INPUT_SHAPE, bucket=BUCKET, batch_size=652, shuffle=False, repeat=1, split="test"
).get_dataset()

splits = {'Train Split': train_batcher, 'Validation Split': val_batcher, 'Test Split': test_batcher}