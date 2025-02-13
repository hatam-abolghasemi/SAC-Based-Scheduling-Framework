import sqlite3

# Connect to SQLite database (this will create jobs.db if it doesn't exist)
conn = sqlite3.connect('jobs.db')
cursor = conn.cursor()

# Create the jobs table
cursor.execute('''
CREATE TABLE IF NOT EXISTS jobs (
    job_id INTEGER PRIMARY KEY,
    dl_batch_size INTEGER,
    dl_learning_rate REAL,
    dl_expected_time INTEGER,
    dl_requested_cpu INTEGER,
    dl_requested_mem TEXT,
    dl_requested_gpu INTEGER,
    dl_dataset TEXT,
    dl_framework TEXT,
    dl_model TEXT
)
''')

# Insert 16 job samples into the database
job_samples = [
    (32, 0.001, 3600, 4, '16Gi', 1, 'ImageNet', 'TensorFlow', 'ResNet50'),
    (64, 0.0005, 7200, 8, '32Gi', 2, 'CIFAR-10', 'PyTorch', 'VGG16'),
    (128, 0.0001, 1800, 2, '8Gi', 1, 'MNIST', 'TensorFlow', 'LeNet'),
    (256, 0.002, 5400, 4, '16Gi', 2, 'CIFAR-100', 'PyTorch', 'ResNet34'),
    (16, 0.001, 3600, 2, '8Gi', 0, 'Fashion-MNIST', 'TensorFlow', 'MLP'),
    (32, 0.0015, 4200, 4, '12Gi', 1, 'COCO', 'PyTorch', 'FasterRCNN'),
    (64, 0.0003, 8000, 6, '24Gi', 2, 'VOC', 'TensorFlow', 'InceptionV3'),
    (128, 0.0002, 3000, 4, '16Gi', 1, 'ADE20K', 'PyTorch', 'UNet'),
    (64, 0.001, 5000, 4, '16Gi', 1, 'Cityscapes', 'TensorFlow', 'SegNet'),
    (128, 0.0008, 10000, 4, '32Gi', 2, 'Oxford Pets', 'PyTorch', 'ResNet101'),
    (32, 0.0012, 4500, 4, '16Gi', 1, 'SUN397', 'TensorFlow', 'DenseNet'),
    (256, 0.0005, 7500, 8, '64Gi', 2, 'Places365', 'PyTorch', 'InceptionV4'),
    (16, 0.002, 3000, 2, '8Gi', 1, 'Flickr', 'TensorFlow', 'AlexNet'),
    (64, 0.0007, 6000, 6, '24Gi', 2, 'Tiny ImageNet', 'PyTorch', 'ResNet152'),
    (32, 0.0018, 8000, 4, '12Gi', 1, 'UCF101', 'TensorFlow', 'Xception'),
    (128, 0.001, 4000, 4, '16Gi', 2, 'Kinetics', 'PyTorch', 'VGG19')
]

# Insert job samples into the database
cursor.executemany('''
INSERT INTO jobs (dl_batch_size, dl_learning_rate, dl_expected_time, dl_requested_cpu, dl_requested_mem, dl_requested_gpu, dl_dataset, dl_framework, dl_model)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
''', job_samples)

# Commit and close the connection
conn.commit()
conn.close()

