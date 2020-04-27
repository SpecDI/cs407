import tensorflow as tf
from tensorflow.keras.models import load_model

import seaborn as sns
import matplotlib.pyplot as plt
# sys.path.append('./action_recognition/architectures')

# Constant variables
FRAME_LENGTH = 80
FRAME_WIDTH = 80
FRAME_NUM = 64

CHANNELS = 3
CLASSES = 13

KERNEL_SHAPE = (3, 3)
POOL_SHAPE = (2, 2)
INPUT_SHAPE = (FRAME_NUM, FRAME_LENGTH, FRAME_WIDTH, CHANNELS)

# Action indices
actions_header = sorted(['Unknown', 'Sitting', 'Lying', 'Drinking', 'Calling', 'Reading', 'Handshaking', 'Running', 'Pushing/Pulling', 'Walking', 'Hugging', 'Carrying', 'Standing'])

def parse_args():
    """ Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Main detection, tracking and action recognition pipeline")

   parser.add_argument(
        "--weights_file", help="Name of weight file to be loaded for the action recognition model",
        required = True)

    return parser

def confusion_matrix(weights_file, test_dir):
    model = load_model(weights_file)

    datagen = ImageDataGenerator(preprocessing_function=self.preprocessing_function)
    test_data = datagen.flow_from_directory(test_dir,
                                        target_size=(FRAME_LENGTH, FRAME_WIDTH),
                                        batch_size=32,
                                        frames_per_step=FRAME_NUM, shuffle=True)

    print("Starting Predictions")
    y_pred = model.predict_generator(test_data, steps=test_data.samples//batch_size+1, verbose=1)

    print('Multi-label confusion matrix')
    y_true = test_data.classes
    matrices = multilabel_confusion_matrix(y_true, np.round(y_pred))

    rows = 5
    cols = 3
    fig, axes = plt.subplots(rows, cols)

    for i, action in enumerate(actions_header):
        matrix = matrices[i]

        row = i // cols
        col = i % cols

        if i % cols == 0:
            row -= 1
            col = cols -1

        sns.heatmap(matrix, annot=True, ax = axes[row, col]); 
        axes[row, col].set_title(action)

    plt.savefig('figures/Multi_Label_Confusion_Matrices.png')


if __name__ == '__main__':
    # Parse user provided arguments
    parser = parse_args()
    args = parser.parse_args()

    confusion_matrix(args.weights_file)