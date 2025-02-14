import sqlite3

# Connect to SQLite database (this will create jobs.db if it doesn't exist)
conn = sqlite3.connect('jobs.db')
cursor = conn.cursor()

# Create the jobs table without dl_expected_time, dl_requested_cpu, dl_requested_mem, dl_requested_gpu
cursor.execute('''
CREATE TABLE IF NOT EXISTS jobs (
    job_id INTEGER PRIMARY KEY,
    dl_batch_size INTEGER,
    dl_learning_rate REAL,
    dl_dataset TEXT,
    dl_framework TEXT,
    dl_model TEXT
)
''')

# Insert 16 job samples into the database without the removed fields
job_samples = [
    (32, 0.001, 'ImageNet', 'TensorFlow', 'ResNet50'),
    (64, 0.0005, 'CIFAR-10', 'PyTorch', 'VGG16'),
    (128, 0.0001, 'MNIST', 'TensorFlow', 'LeNet'),
    (256, 0.002, 'CIFAR-100', 'PyTorch', 'ResNet34'),
    (16, 0.001, 'Fashion-MNIST', 'TensorFlow', 'MLP'),
    (32, 0.0015, 'COCO', 'PyTorch', 'FasterRCNN'),
    (64, 0.0003, 'VOC', 'TensorFlow', 'InceptionV3'),
    (128, 0.0002, 'ADE20K', 'PyTorch', 'UNet'),
    (64, 0.001, 'Cityscapes', 'TensorFlow', 'SegNet'),
    (128, 0.0008, 'Oxford Pets', 'PyTorch', 'ResNet101'),
    (32, 0.0012, 'SUN397', 'TensorFlow', 'DenseNet'),
    (256, 0.0005, 'Places365', 'PyTorch', 'InceptionV4'),
    (16, 0.002, 'Flickr', 'TensorFlow', 'AlexNet'),
    (64, 0.0007, 'Tiny ImageNet', 'PyTorch', 'ResNet152'),
    (32, 0.0018, 'UCF101', 'TensorFlow', 'Xception'),
    (128, 0.001, 'Kinetics', 'PyTorch', 'VGG19')
]

# Insert job samples into the database without the removed fields
cursor.executemany('''
INSERT INTO jobs (dl_batch_size, dl_learning_rate, dl_dataset, dl_framework, dl_model)
VALUES (?, ?, ?, ?, ?)
''', job_samples)

# Commit and close the connection
conn.commit()
conn.close()

