# evaluate the model on the test set
import os
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
from train import build_3dcnn_model, build_efficientnet_model, CFG
from train import create_pipeline
import config as CFG

def plot_confusion_matrix(y_true, y_pred, TEN_LABEL_LIST, result_folder):
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Create figure and axes with larger size
    plt.figure(figsize=(10, 10))
    # Plot heatmap with improved formatting
    sns.heatmap(cm, 
                annot=True,           # Show numbers in cells
                fmt='d',             # Use integer format
                cmap='Blues',        # Color scheme
                xticklabels=TEN_LABEL_LIST,
                yticklabels=TEN_LABEL_LIST)
    # Add labels and title with better visibility
    plt.xlabel('Predicted Label', fontsize=12, labelpad=10)
    plt.ylabel('True Label', fontsize=12, labelpad=10)
    plt.title('Confusion Matrix', fontsize=14, pad=20)
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=45, ha='right')
    # Save and close
    plt.savefig(result_folder + 'confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

def evaluate_model(model, test_ds, test_df, TEN_LABEL_LIST, result_folder):
    # Compile the model first
    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        metrics=['accuracy']
    )
    
    # # Calculate steps properly, rounding up to include all samples
    # total_steps = tf.math.ceil(len(test_df) / CFG.BATCH_SIZE)
    
    # Evaluate the model
    test_loss, test_acc = model.evaluate(
        test_ds,
        # steps=total_steps,
        verbose=1
    )

    # Get predictions for all samples
    y_true = test_df['label_encoded'].to_list()
    y_pred_prob = model.predict(
        test_ds,
        # steps=total_steps,
        verbose=1
    )
    
    # Only take predictions for the actual number of samples
    y_pred = np.argmax(y_pred_prob[:len(y_true)], axis=1)
    print(f"Test loss: {test_loss:.4f}, Test accuracy: {test_acc:.4f}")
    # Plot confusion matrix
    plot_confusion_matrix(y_true, y_pred, TEN_LABEL_LIST, result_folder)

if __name__ == "__main__":

    result_folder = 'results/' + CFG.MODEL_TYPE + '_' + str(CFG.NUM_CLASSES) + '/'

    train_df = pd.read_csv(CFG.TRAIN_CSV)
    label_distribution = train_df['label'].value_counts()
    TEN_LABEL_LIST = list(label_distribution.keys())[:CFG.NUM_CLASSES]
    
    # Load the test set
    test_df = pd.read_csv(CFG.TEST_CSV)
    test_df['clip_path'] = test_df['clip_path'].apply(lambda x: os.path.join(CFG.DATASET_PATH, x.lstrip('/')))
    test_df = test_df[test_df['label'].isin(TEN_LABEL_LIST)]
    test_df['label_encoded'] = test_df['label'].apply(lambda x: TEN_LABEL_LIST.index(x))

    # Create the test dataset
    test_ds = create_pipeline(test_df, batch_size=CFG.BATCH_SIZE, is_training=False)

    # Load the model
    if CFG.MODEL_TYPE == '3dcnn':
        model = build_3dcnn_model()
    elif CFG.MODEL_TYPE == 'efficientnet':
        model = build_efficientnet_model()
    else:
        raise ValueError(f"Invalid model type: {CFG.MODEL_TYPE}")
    
    model.load_weights(f'{result_folder}/{CFG.MODEL_TYPE}_MODEL_{CFG.NUM_CLASSES}.weights.h5')
    model.summary()

    # Evaluate the model
    evaluate_model(model, test_ds, test_df, TEN_LABEL_LIST, result_folder)