import os
import pandas as pd
import numpy as np
import sys
#from tabulate import tabulate
import cv2 as cv

# Const variables
GOLD_BB = 'bounding_boxes'
PREDS_BB = 'box_preds'
GOLD_SS = 'masks'
PREDS_SS = 'mask_preds'
IOU_THRESH = 0.5

def process_bounding_boxes(relative_path: str, folder: str) -> pd.DataFrame:
    '''
        parameters
            relative_path, path to the root folder of the dataset
            folder, name of the folder to process

        returns
            pd.DataFrame with the data of all files in the folder
    '''
    df = pd.DataFrame(columns=['File','ID', 'X1', 'Y1', 'width', 'heigth'])
    file_counter = 0

    for root, dirs, files in os.walk(relative_path):
        if folder in dirs:
            bounding_box_dir = os.path.join(root, folder)

            for filename in os.listdir(bounding_box_dir):
                if filename.endswith('.txt'):
                    file_path = os.path.join(bounding_box_dir, filename)
                    data = load_bounding_box_data(file_path, file_counter)

                    if data is not None:
                        df = df.append(data, ignore_index=True)
                        file_counter += 1

    return df

def load_bounding_box_data(file_path: str, file_counter: int) -> pd.DataFrame:
    '''
        parameters
            file_path, full path to a .txt file

        returns
          pd.DataFrame with the data stored into a file
    '''
    with open(file_path, 'r') as file:
        lines = file.readlines()

    data = []
    for line in lines:
        line = line.strip()
        if line:
            parts = line.split(';')
            id_, coordinates = parts[0].strip(), parts[1].strip()[1:-1].split(', ')
            id_ = int(id_[3:])
            coordinates = [int(coord) for coord in coordinates]
            row = [file_counter, id_] + coordinates
            data.append(row)

    if data:
        return pd.DataFrame(data, columns=['File','ID', 'X1', 'Y1', 'width', 'heigth'])
    else:
        return None

def process_box_preds(result_df: pd.DataFrame, pred_df: pd.DataFrame, confusion_df: pd.DataFrame) -> float:
    '''
        parameters
            result_df, dataframe with the ground truth
            pred_df, dataframe with the predictions
            confusion_df, dataframe with the confusion matrix

        returns
            mAP, mean Average Precision
    '''
    #For each File of the dataframe, range over the IDs and compute IoU
    for file in result_df['File'].unique():
        for id in result_df.loc[result_df['File'] == file]['ID'].unique():
            result = result_df.loc[(result_df['File'] == file) & (result_df['ID'] == id)]
            pred = pred_df.loc[(pred_df['File'] == file) & (pred_df['ID'] == id)]

            # If there is no prediction, it is a FN
            if pred.empty:
                confusion_df.loc[confusion_df['ID'] == id, 'FN'] += 1
            else:
                # Compute IoU
                xA = max(result['X1'].values[0], pred['X1'].values[0])
                yA = max(result['Y1'].values[0], pred['Y1'].values[0])
                xB = min(result['X1'].values[0] + result['width'].values[0],\
                         pred['X1'].values[0] + pred['width'].values[0])
                yB = min(result['Y1'].values[0] + result['heigth'].values[0],\
                         pred['Y1'].values[0] + pred['heigth'].values[0])

                interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

                resultArea = (result['width'].values[0] + 1) * (result['heigth'].values[0] + 1)
                predArea = (pred['width'].values[0] + 1) * (pred['heigth'].values[0] + 1)

                iou = interArea / float(resultArea + predArea - interArea)

                # If IoU >= thresh, it is a TP, else it is a FP
                if iou >= IOU_THRESH:
                    confusion_df.loc[confusion_df['ID'] == id, 'TP'] += 1
                else:
                    confusion_df.loc[confusion_df['ID'] == id, 'FP'] += 1
                    confusion_df.loc[confusion_df['ID'] == id, 'FN'] += 1    

    # Compute Precision-Recall for each class
    for id in confusion_df['ID'].unique():
        TP = confusion_df.loc[confusion_df['ID'] == id, 'TP']
        FP = confusion_df.loc[confusion_df['ID'] == id, 'FP']
        FN = confusion_df.loc[confusion_df['ID'] == id, 'FN']

        confusion_df['ID']['precision']= TP / (TP + FP)
        confusion_df['ID']['recall'] = TP / (TP + FN)

    # Compute AP for each class
    for id in confusion_df['ID'].unique():
        precision = confusion_df.loc[confusion_df['ID'] == id, 'precision']
        recall = confusion_df.loc[confusion_df['ID'] == id, 'recall']

        confusion_df['ID']['AP'] = np.trapz(precision, recall)

    # Compute mAP
    print(confusion_df['AP'])
    mAP = confusion_df['AP'].mean()
    return mAP

def load_semantic_segmentation_data(file_path: str):
    '''
        parameters
            file_path, full path to a .png file

        returns
            cv.Mat with the data stored into a file
    '''
    return cv.imread(file_path, cv.IMREAD_GRAYSCALE)


def process_semantic_segmentation(relative_path: str, folder: str) -> list:
    images = []
    for root, dirs, files in os.walk(relative_path):
        if folder in dirs:
            bounding_box_dir = os.path.join(root, folder)

            for filename in os.listdir(bounding_box_dir):
                if filename.endswith('.png'):
                    file_path = os.path.join(bounding_box_dir, filename)
                    data = load_semantic_segmentation_data(file_path)

                    if data is not None:
                       images.append(data)

    return images


def main():
    assert len(sys.argv) == 2, "Usage: python Eval.py <relative_path>"
    relative_path = sys.argv[1]

# Bounding Boxes evaluation
    # Create the dataframes with the results
    result_df = process_bounding_boxes(relative_path, GOLD_BB)
    pred_df = process_bounding_boxes(relative_path, GOLD_BB)

    # Create the Confusion Matrix
    data = { 'ID': np.arange(14),
             'TP': [0]*14,
             'FP': [0]*14,
             'FN': [0]*14,
             'TN': [0]*14,
             'precision': [0]*14,
             'recall': [0]*14,
             'AP': [0]*14
        }

    confusion_df = pd.DataFrame(data)
    # Compute mAP
    mAP = process_box_preds(result_df, pred_df, confusion_df)
    print('mAP: ', mAP)

#  Semantic Segmentation evaluation
    # Load the images
    result_images = process_semantic_segmentation(relative_path, GOLD_SS)
    pred_images = process_semantic_segmentation(relative_path, PREDS_SS)

    # Compute the IoU
    

# Print results


if __name__ == "__main__":
	main()
