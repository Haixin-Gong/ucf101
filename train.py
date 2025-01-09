import random
import os
import glob
import time

import pandas as pd
import numpy as np
import cv2
import config as CFG
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers


def preprocess_frame(frame, frame_size):
    processed_frame = tf.cast(frame, tf.float32) / 255.0
    # Get current dimensions
    height, width = frame.shape[0], frame.shape[1]
    target_height, target_width = frame_size
    
    # Use resize instead of resize_with_pad for more stability
    processed_frame = tf.image.resize(
        processed_frame,
        size=(target_height, target_width),
        method=tf.image.ResizeMethod.BILINEAR
    )
    
    # Ensure the output shape is correct
    processed_frame.set_shape((target_height, target_width, 3))
    
    return processed_frame

def get_frames(avi_path, frame_count=15, frame_size=(224,224), frame_step=5):
    """
    Extract frames from a video file with specific sampling strategy.
    
    Args:
        avi_path (str): Path to the video file
        frame_count (int): Number of frames to extract
        frame_size (tuple): Target size for frames (height, width)
        frame_step (int): Number of frames to skip between each sampled frame
    """
    start = 0
    frames = []
    
    if not os.path.exists(str(avi_path)):
        print(f"Error: Video file not found: {avi_path}")
        return None
    
    source = cv2.VideoCapture(str(avi_path))
    if not source.isOpened():
        print(f"Error: Could not open video file: {avi_path}")
        return None
    
    # Calculate required video length and random start point
    clip_length = source.get(cv2.CAP_PROP_FRAME_COUNT)  # Total frames in video
    required_length = 1 + (frame_step * (frame_count - 1))  # Frames needed for sampling
    
    # If video is long enough, choose random starting point
    if required_length < clip_length:
        max_start = clip_length - required_length
        start = random.randint(0, max_start - 1)
    
    # Set video to starting frame
    source.set(cv2.CAP_PROP_POS_FRAMES, start)
    ret, frame = source.read()
    if not ret:
        frame = np.zeros((*frame_size, 3), dtype=np.float32)
        frames.append(frame)
    else:
        processed_frame = preprocess_frame(frame, frame_size)
        frames.append(processed_frame)

    for i in range(frame_count - 1):
        for _ in range(frame_step):
            ret, frame = source.read()

        if not ret:
            frames.append(np.zeros((*frame_size, 3), dtype=np.float32))
        else:
            processed_frame = preprocess_frame(frame, frame_size)
            frames.append(processed_frame)
    # Clean up video capture
    source.release()
    # Convert to numpy array and reorder channels from BGR to RGB
    frames = np.array(frames)
    frames = frames[..., [2, 1, 0]] # BGR to RGB
    assert frames.shape == (frame_count, *frame_size, 3)
    return frames

class ClipGenerator:
    def __init__(self, clip_paths, clip_labels, frame_count, frame_size=(224, 224), frame_step=15):
        self.clip_paths = clip_paths
        self.clip_labels = clip_labels
        self.frame_count = frame_count
        self.frame_size = frame_size 
        self.frame_step = frame_step
        
    def __call__(self):
        data = list(zip(self.clip_paths, self.clip_labels))
        for path, label in data:
            clip_frames = get_frames(path, 
                                   frame_count=self.frame_count, 
                                   frame_size=self.frame_size, 
                                   frame_step=self.frame_step)
            if clip_frames is not None:
                yield clip_frames, label
            else:
                print(f"Skipping problematic video: {path}")
                continue

def create_pipeline(df, batch_size=32, shuffle=False, cache=False, prefetch=False, is_training=False):
    clip_paths = df['clip_path'].to_list()
    clip_labels = tf.one_hot(df.label_encoded, depth=CFG.NUM_CLASSES).numpy()
    AUTOTUNE = tf.data.AUTOTUNE
    
    gen_output_signature = (
        tf.TensorSpec(shape=CFG.CLIP_SIZE, dtype=tf.float32), 
        tf.TensorSpec(shape=(CFG.NUM_CLASSES,), dtype=tf.float32)
    )
    
    ds = tf.data.Dataset.from_generator(
        ClipGenerator(clip_paths, clip_labels, CFG.FRAME_COUNT, CFG.FRAME_SIZE, CFG.FRAME_STEP),
        output_signature=gen_output_signature
    )
    
    if shuffle:
        ds = ds.shuffle(buffer_size=1000)
    if is_training:
        ds = ds.repeat()  # Only repeat for training
    ds = ds.batch(batch_size)
    if cache:
        ds = ds.cache()
    if prefetch:
        ds = ds.prefetch(buffer_size=AUTOTUNE)
    
    return ds

def build_3dcnn_model():
    """
    Build a lightweight 3D CNN model for video classification
    """
    inputs = layers.Input(shape=CFG.CLIP_SIZE, dtype=tf.float32)
    x = layers.Rescaling(scale=255)(inputs)
    
    # First 3D Conv block
    x = layers.Conv3D(32, kernel_size=(3,3,3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool3D(pool_size=(1,2,2), strides=(1,2,2))(x)
    # Output: (10, 112, 112, 32)
    
    # Second 3D Conv block
    x = layers.Conv3D(64, kernel_size=(3,3,3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool3D(pool_size=(2,2,2), strides=(2,2,2))(x)
    # Output: (5, 56, 56, 64)
    
    # Third 3D Conv block
    x = layers.Conv3D(128, kernel_size=(3,3,3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool3D(pool_size=(2,2,2), strides=(2,2,2))(x)
    # Output: (2, 28, 28, 128)
    
    # Global pooling and dense layers
    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(CFG.NUM_CLASSES, activation='softmax')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="lightweight_3DCNN")
    return model

def build_efficientnet_model():
    initializer = tf.keras.initializers.GlorotNormal()
    clip_input = layers.Input(shape=CFG.CLIP_SIZE, dtype=tf.float32, name='clip_input')
    
    rescale = layers.Rescaling(scale=255)(clip_input)
    
    efficientnet_v1_b0 = tf.keras.applications.efficientnet.EfficientNetB0(
        input_shape=(CFG.WIDTH, CFG.HEIGHT, CFG.CHANNELS), 
        include_top=False
    )
    efficientnet_v1_b0.trainable = False
    
    time_distributed_layer = layers.TimeDistributed(efficientnet_v1_b0)(rescale)
    output_layer = layers.Dense(64, activation='relu', kernel_initializer=initializer)(time_distributed_layer)
    output_layer = layers.GlobalAveragePooling3D()(output_layer)
    output_layer = layers.Dense(CFG.NUM_CLASSES, activation='softmax', kernel_initializer=initializer)(output_layer)
    # output_layer = layers.GlobalAveragePooling3D()(output_layer)

    model = tf.keras.Model(inputs=[clip_input], outputs=[output_layer], 
                         name="multiclass_action_recognition_model")
    return model

def main():
    # GPU Configuration
    if CFG.DEVICE == 'GPU':
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if not gpus:
            raise RuntimeError("No GPU found. Please check your CUDA installation.")
        
        try:
            # Configure GPU memory growth
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            # Verify GPU is being used
            with tf.device('/GPU:0'):
                # Simple test tensor operation
                a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
                b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
                c = tf.matmul(a, b)
            
            print(f"\nGPU setup successful. Found {len(gpus)} GPU(s):")
            for gpu in gpus:
                print(f"  {gpu}")
                
        except RuntimeError as e:
            print(f"GPU setup failed: {e}")
            raise RuntimeError("GPU initialization failed")
    else:
        print("Training on CPU (DEVICE set to 'CPU' in config)")
        
    # Print detailed device info
    print("\nDevice Strategy:", tf.distribute.get_strategy())
    print("TensorFlow version:", tf.__version__)
    print("CUDA available:", tf.test.is_built_with_cuda())
    
    try:
        print("\nGPU Memory Details:")
        print(tf.config.experimental.get_memory_info('GPU:0'))
    except:
        pass
    
    # Debug: Check if files exist
    for path in [CFG.TRAIN_CSV, CFG.VAL_CSV, CFG.TEST_CSV]:
        if not os.path.exists(path):
            print(f"Error: File not found: {path}")
            return
            
    # Set random seeds
    tf.random.set_seed(CFG.TF_SEED)
    
    # Load and prepare data
    train_df = pd.read_csv(CFG.TRAIN_CSV)
    val_df = pd.read_csv(CFG.VAL_CSV)
    test_df = pd.read_csv(CFG.TEST_CSV)

    result_folder = 'results/' + CFG.MODEL_TYPE + '_' + str(CFG.NUM_CLASSES) + '/'
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    # View Label Distibutions
    plt.figure(figsize=(15, 20))
    plt.title('Train Labels Distribution', fontsize=15)
    label_distribution = train_df['label'].value_counts()
    sns.barplot(x=label_distribution.values,
                y=list(label_distribution.keys()),
                width=0.8,
                orient="h")
    plt.savefig(result_folder + 'train_label_distribution.png', dpi=300)
    plt.close()

    # complete the clip_path
    train_df['clip_path'] = train_df['clip_path'].apply(lambda x: os.path.join(CFG.DATASET_PATH, x.lstrip('/')))
    val_df['clip_path'] = val_df['clip_path'].apply(lambda x: os.path.join(CFG.DATASET_PATH, x.lstrip('/')))
    test_df['clip_path'] = test_df['clip_path'].apply(lambda x: os.path.join(CFG.DATASET_PATH, x.lstrip('/')))
    
    # Create binary datasets
    TEN_LABEL_LIST = list(label_distribution.keys())[:CFG.NUM_CLASSES]
    train_binary_df = train_df[train_df['label'].isin(TEN_LABEL_LIST)].sample(frac=1.0).reset_index(drop=True)
    val_binary_df = val_df[val_df['label'].isin(TEN_LABEL_LIST)].sample(frac=1.0).reset_index(drop=True)
    test_binary_df = test_df[test_df['label'].isin(TEN_LABEL_LIST)].sample(frac=1.0).reset_index(drop=True)
    
    # Encode labels
    encode_label = lambda label: TEN_LABEL_LIST.index(label)
    for df in [train_binary_df, val_binary_df, test_binary_df]:
        df['label_encoded'] = df['label'].apply(encode_label)

    # Create data pipelines
    train_ds = create_pipeline(train_binary_df, batch_size=CFG.BATCH_SIZE, prefetch=True, is_training=True)
    val_ds = create_pipeline(val_binary_df, batch_size=CFG.BATCH_SIZE, is_training=True)
    
    # Build and compile model
    if CFG.MODEL_TYPE == '3dcnn':
        model = build_3dcnn_model()
    elif CFG.MODEL_TYPE == 'efficientnet':
        model = build_efficientnet_model()
    else:
        raise ValueError(f"Invalid model type: {CFG.MODEL_TYPE}")

    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        optimizer=tf.keras.optimizers.Adam(learning_rate=CFG.LEARNING_RATE),
        metrics=['accuracy']
    )

    model.summary()
    # Define callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', 
            patience=3, 
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            patience=2,
            factor=0.1,
            verbose=1
        )
    ]
    
    # Add time tracking
    start_time = time.time()
    
    # Train model
    print(f'Training {model.name}')
    print(f'Train on {len(train_binary_df)} samples, validate on {len(val_binary_df)} samples.')
    
    # Calculate exact steps
    steps_per_epoch = len(train_binary_df) // CFG.BATCH_SIZE
    validation_steps = len(val_binary_df) // CFG.BATCH_SIZE
    
    # Train model with explicit steps
    history = model.fit(
        train_ds,
        epochs=CFG.EPOCHS,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_ds,
        validation_steps=validation_steps,
        callbacks=callbacks
    )
    
    # Calculate training time
    end_time = time.time()
    training_time = end_time - start_time
    hours = int(training_time // 3600)
    minutes = int((training_time % 3600) // 60)
    seconds = int(training_time % 60)
    
    print(f'Training completed in {len(history.history["loss"])} epochs')
    print(f"\nTraining completed in {hours:02d}:{minutes:02d}:{seconds:02d}")
    
    # plot the training loss and accuracy
    plt.figure(figsize=(10, 10))
    plt.subplot(2, 1, 1)
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.xticks(range(0, len(history.history['loss']), 2))
    plt.title(f'Training History (Time: {hours:02d}:{minutes:02d}:{seconds:02d})')
    plt.legend(loc='upper right')
    
    # plot the training accuracy and validation accuracy
    plt.subplot(2, 1, 2)
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='val')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.xticks(range(0, len(history.history['loss']), 2))
    plt.legend(loc='upper right')
    plt.savefig(result_folder + 'train_loss_accuracy.png', dpi=300)
    plt.close()
    # save the model
    model.save_weights(result_folder + CFG.MODEL_TYPE + '_MODEL_' + str(CFG.NUM_CLASSES) + '.weights.h5')

if __name__ == "__main__":
    main()