from keras import layers
import keras

import matplotlib.pyplot as plt
import numpy as np
import configparser

# Parse config.ini --------------------------------------------------------------
config = configparser.ConfigParser()
config.read('config.ini')

# Hyperparameters and constants --------------------------------------------------------------
# METADATA Section
framework = config.get('METADATA', 'framework')
model = config.get('METADATA', 'model')
dataset = config.get('METADATA', 'dataset')
field = config.get('METADATA', 'field')
subfield = config.get('METADATA', 'subfield')

# GENERAL Section
learning_rate = config.getfloat('GENERAL', 'learning_rate')
weight_decay = config.getfloat('GENERAL', 'weight_decay')
batch_size = config.getint('GENERAL', 'batch_size')
num_epochs = config.getint('GENERAL', 'num_epochs')
units = list(map(int, config.get('GENERAL', 'units').split(',')))
num_layers = config.getint('GENERAL', 'layers')

# COMPUTER_VISION Section
input_shape = tuple(map(int, config.get('COMPUTER_VISION', 'input_shape').split(',')))
image_size = config.getint('COMPUTER_VISION', 'image_size')
conv_layers = config.getint('COMPUTER_VISION', 'conv_layers')
projection_dim = config.getint('COMPUTER_VISION', 'projection_dim')
num_heads = config.getint('COMPUTER_VISION', 'num_heads')
positional_emb = config.getboolean('COMPUTER_VISION', 'positional_emb')
stochastic_depth_rate = config.getfloat('COMPUTER_VISION', 'stochastic_depth_rate')

# IMAGE_CLASSIFICATION Section
num_classes = config.getint('IMAGE_CLASSIFICATION', 'num_classes')
transformer_units = list(map(int, config.get('IMAGE_CLASSIFICATION', 'transformer_units').split(',')))
transformer_layers = config.getint('IMAGE_CLASSIFICATION', 'transformer_layers')

# Load MNIST dataset --------------------------------------------------------------
num_classes = 10
input_shape = (28, 28, 1)

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Reshape and normalize data for the model
x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255
x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

print(f"x_train shape: {x_train.shape} - y_train shape: {y_train.shape}")
print(f"x_test shape: {x_test.shape} - y_test shape: {y_test.shape}")

# Update the data augmentation pipeline
data_augmentation = keras.Sequential(
    [
        layers.Rescaling(scale=1.0 / 255),
        layers.RandomZoom(0.1),
    ],
    name="data_augmentation",
)

# The final CCT model --------------------------------------------------------------
def create_cct_model(
    image_size=image_size,
    input_shape=input_shape,
    num_heads=num_heads,
    projection_dim=projection_dim,
    transformer_units=transformer_units,
):
    inputs = layers.Input(input_shape)

    # Augment data.
    augmented = data_augmentation(inputs)

    # Encode patches.
    cct_tokenizer = CCTTokenizer()
    encoded_patches = cct_tokenizer(augmented)

    # Apply positional embedding.
    if positional_emb:
        sequence_length = encoded_patches.shape[1]
        encoded_patches += PositionEmbedding(sequence_length=sequence_length)(
            encoded_patches
        )

    # Calculate Stochastic Depth probabilities.
    dpr = [x for x in np.linspace(0, stochastic_depth_rate, transformer_layers)]

    # Create multiple layers of the Transformer block.
    for i in range(transformer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-5)(encoded_patches)

        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)

        # Skip connection 1.
        attention_output = StochasticDepth(dpr[i])(attention_output)
        x2 = layers.Add()([attention_output, encoded_patches])

        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-5)(x2)

        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)

        # Skip connection 2.
        x3 = StochasticDepth(dpr[i])(x3)
        encoded_patches = layers.Add()([x3, x2])

    # Apply sequence pooling.
    representation = layers.LayerNormalization(epsilon=1e-5)(encoded_patches)
    weighted_representation = SequencePooling()(representation)

    # Classify outputs.
    logits = layers.Dense(num_classes)(weighted_representation)
    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=logits)
    return model

# Model training and evaluation --------------------------------------------------------------
def run_experiment(model):
    optimizer = keras.optimizers.AdamW(learning_rate=0.001, weight_decay=0.0001)

    model.compile(
        optimizer=optimizer,
        loss=keras.losses.CategoricalCrossentropy(
            from_logits=True, label_smoothing=0.1
        ),
        metrics=[
            keras.metrics.CategoricalAccuracy(name="accuracy"),
            keras.metrics.TopKCategoricalAccuracy(5, name="top-5-accuracy"),
        ],
    )

    checkpoint_filepath = "/tmp/checkpoint.weights.h5"
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        checkpoint_filepath,
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=True,
    )

    history = model.fit(
        x=x_train,
        y=y_train,
        batch_size=batch_size,
        epochs=num_epochs,
        validation_split=0.1,
        callbacks=[checkpoint_callback],
    )

    model.load_weights(checkpoint_filepath)
    _, accuracy, top_5_accuracy = model.evaluate(x_test, y_test)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")
    print(f"Test top 5 accuracy: {round(top_5_accuracy * 100, 2)}%")

    return history


cct_model = create_cct_model()
history = run_experiment(cct_model)

plt.plot(history.history["loss"], label="train_loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Train and Validation Losses Over Epochs", fontsize=14)
plt.legend()
plt.grid()
plt.show()
