import sqlite3

# Connect to SQLite database (this will create jobs.db if it doesn't exist)
conn = sqlite3.connect('jobs.db')
cursor = conn.cursor()

# Create the jobs table
cursor.execute('''
CREATE TABLE IF NOT EXISTS jobs (
    job_id INTEGER PRIMARY KEY,
    job_batch_size INTEGER,
    job_learning_rate REAL,
    job_dataset_complexity REAL,
    job_model_complexity REAL
);
''')

# Job samples — 32 rows matching the SQL script
job_samples = [
    (32, 0.001, 7.5, 8.2),  # ImageNet, ResNet50
    (64, 0.0005, 5.0, 7.0),  # CIFAR-10, VGG16
    (128, 0.0001, 3.5, 4.5), # MNIST, LeNet
    (256, 0.002, 6.0, 7.8),  # CIFAR-100, ResNet34
    (16, 0.001, 2.5, 3.0),   # Fashion-MNIST, MLP
    (32, 0.0015, 8.0, 9.5),  # COCO, FasterRCNN
    (64, 0.0003, 4.2, 5.6),  # VOC, InceptionV3
    (128, 0.0002, 7.0, 8.8), # ADE20K, UNet
    (64, 0.001, 6.5, 7.2),   # Cityscapes, SegNet
    (128, 0.0008, 5.8, 6.5), # Oxford Pets, ResNet101
    (32, 0.0012, 3.8, 4.8),  # SUN397, DenseNet
    (256, 0.0005, 9.0, 9.0), # Places365, InceptionV4
    (16, 0.002, 2.0, 3.2),   # Flickr, AlexNet
    (64, 0.0007, 7.2, 8.0),  # Tiny ImageNet, ResNet152
    (32, 0.0018, 5.5, 6.0),  # UCF101, Xception
    (128, 0.001, 4.0, 5.0),  # Kinetics, VGG19

    # Additional 16 jobs
    (128, 0.0015, 8.5, 9.0), # ImageNet, ResNet101
    (256, 0.0006, 4.5, 7.5), # CIFAR-10, VGG19
    (64, 0.0004, 3.0, 4.0),  # MNIST, MLP
    (32, 0.002, 6.2, 7.0),   # CIFAR-100, ResNet18
    (128, 0.0007, 2.8, 3.5), # Fashion-MNIST, SVM
    (256, 0.0012, 8.3, 9.2), # COCO, MaskRCNN
    (32, 0.0002, 4.0, 5.8),  # VOC, EfficientNet
    (64, 0.0011, 7.2, 8.0),  # ADE20K, DeepLabV3
    (128, 0.0009, 6.8, 7.5), # Cityscapes, PSPNet
    (256, 0.0003, 6.0, 6.8), # Oxford Pets, DenseNet201
    (32, 0.0014, 3.5, 4.5),  # SUN397, GoogLeNet
    (64, 0.0008, 9.5, 9.8),  # Places365, NASNet
    (128, 0.002, 2.2, 3.0),  # Flickr, SqueezeNet
    (32, 0.0006, 7.5, 8.5),  # Tiny ImageNet, MobileNet
    (256, 0.0013, 5.2, 6.0), # UCF101, ShuffleNet
    (64, 0.0009, 4.2, 5.0)   # Kinetics, AlexNet
]

# Insert job samples into the database
cursor.executemany('''
INSERT INTO jobs (job_batch_size, job_learning_rate, job_dataset_complexity, job_model_complexity)
VALUES (?, ?, ?, ?)
''', job_samples)

# Commit and close the connection
conn.commit()
conn.close()

print("Database populated with job samples!")

