import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score
from scipy.spatial.distance import cosine
from scipy.stats import entropy
import matplotlib.pyplot as plt
import os
import pickle


import torch

import pickle
import torch

def process_label(value):
        """Rounds the value to the nearest integer and clips it within the allowed range."""
        rounded = round(value)
        return max(-3, min(rounded, 3))
def save_comparison_data_pickle(filepath, predictions, targets, masks_txt, masks_au, masks_vi):
    """
    Saves predictions, targets, and masks to a pickle file.

    Args:
        filepath (str): Path to save the pickle file.
        predictions (List[torch.Tensor]): Model predictions.
        targets (List[torch.Tensor]): Ground truth targets.
        masks_txt (List[torch.Tensor]): Text masks.
        masks_au (List[torch.Tensor]): Audio masks.
        masks_vi (List[torch.Tensor]): Visual masks.
    """
    data = {
        'predictions': predictions.cpu().numpy(),  # Concatenate if applicable
        'targets': targets.cpu().numpy(),          # Concatenate if applicable
        'masks_txt': [mask.cpu().numpy() for mask in masks_txt],     # List of NumPy arrays
        'masks_au': [mask.cpu().numpy() for mask in masks_au],       # List of NumPy arrays
        'masks_vi': [mask.cpu().numpy() for mask in masks_vi],       # List of NumPy arrays
    }
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    print(f"Comparison data saved to {filepath}")


def load_comparison_data_pickle(filepath):
    """
    Loads predictions, targets, and masks from a pickle file.

    Args:
        filepath (str): Path to the pickle file.

    Returns:
        dict: A dictionary containing predictions, targets, and masks.
    """
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data

# Usage
#data = load_comparison_data_pickle('comparison_data.pkl')
#predictions = data['predictions']
#targets = data['targets']
#masks_txt = data['masks_txt']
#masks_au = data['masks_au']
#masks_vi = data['masks_vi']



# mosei_metrics.py

def calculate_mask_metrics(mask1, mask2):
    """
    Calculates MAE, MSE, Cosine Similarity, and KL-Divergence between two masks.

    Args:
        mask1 (numpy.ndarray): First mask array.
        mask2 (numpy.ndarray): Second mask array.

    Returns:
        dict: Dictionary containing the calculated metrics.
    """
    metrics = {}
    
    # Ensure masks are flattened for comparison
    mask1_flat = mask1.flatten()
    mask2_flat = mask2.flatten()
    
    # Mean Absolute Error
    mae = np.mean(np.abs(mask1_flat - mask2_flat))
    metrics['MAE'] = mae
    
    # Mean Squared Error
    mse = np.mean((mask1_flat - mask2_flat) ** 2)
    metrics['MSE'] = mse
    
    # Cosine Similarity
    cos_sim = 1 - cosine(mask1_flat, mask2_flat)
    metrics['Cosine_Similarity'] = cos_sim
    
    # KL-Divergence
    # Add a small epsilon to avoid division by zero and log(0)
    epsilon = 1e-10
    mask1_probs = mask1_flat + epsilon
    mask2_probs = mask2_flat + epsilon
    mask1_probs /= np.sum(mask1_probs)
    mask2_probs /= np.sum(mask2_probs)
    kl_div = entropy(mask1_probs, mask2_probs)
    metrics['KL_Divergence'] = kl_div
    
    return metrics

def compare_masks(data_new, data_comparison_link):
    """Function to compare masks between two models."""
    metrics_all = {} #stores each of the 4 metrics per modality
   
    #seperates the masks by the target that their input should have been
    masks_per_target_new = {}
    masks_per_target_comp = {}
    mask_per_pred_new = {}

    #mean value of masks (2d array) per modality
    mean_mask_new = {}

    #the difference of mean between the new data to the comparison data per modality 
    diff_mean_mask_new = {}

    #similarly to before but grouped per target
    mean_mask_new_target = {}
    diff_mean_mask_new_target = {}   

    mean_mask_new_pred = {}

    targets = data_new['targets']
    predictions = data_new['predictions']
    data = load_comparison_data_pickle(data_comparison_link)

    for modality in ["txt", "au", "vi"]:
        print(f"Comparing masks for modality '{modality}'")
        masks_transformed_comp = []
        masks_transformed_new = []
        dict_temp = {}
        dict_temp_comp = {}
        dict_temp_pred = {}

        #get the masks for the new data and the comparison data
        mask_new = data_new.get(f'masks_{modality}', [])
        mask_comparison = data.get(f'masks_{modality}', [])


        # Check if both mask lists have the same length
        if len(mask_new) != len(mask_comparison):
            raise ValueError(
            f"Number of masks for modality '{modality}' does not match between datasets. "
            f"data_new has {len(mask_new)} masks, "
            f"data_comparison has {len(mask_comparison)} masks."
        )

        for i in range(len(mask_new)):
            #calculate the metrics for each mask
            mask_metrics = calculate_mask_metrics(mask_new[i], mask_comparison[i])
            
            # store the metrics for each mask
            for metric_name, metric_value in mask_metrics.items():
                key = f'{modality}_{metric_name}'
                if key not in metrics_all:
                    metrics_all[key] = []
                metrics_all[key].append(metric_value) 
            
            

            # Initialize lists if keys do not exist
            tar = process_label(targets[i])
            pred = process_label(predictions[i])
            if tar not in dict_temp:
               dict_temp[tar] = []
            if tar not in dict_temp_comp:
                dict_temp_comp[tar] = []
            if pred not in dict_temp_pred:
                dict_temp_pred[pred] = []
            
            #flatten them to the feature dimension
            mask_flat_new = np.mean(mask_new[i], axis=(0))
            mask_flat_comp = np.mean(mask_comparison[i], axis=(0))
            #append the flattened masks to the list for mean per modality
            masks_transformed_new.append(mask_flat_new)
            masks_transformed_comp.append(mask_flat_comp)
            #append the flattened masks to the list for mean per modality per target
            dict_temp[tar].append(mask_flat_new)
            dict_temp_comp[tar].append(mask_flat_comp)
            dict_temp_pred[pred].append(mask_flat_new)

        # Compute the mean and difference mask for this modality
        masks_per_target_comp[modality] = dict_temp_comp
        masks_per_target_new[modality] = dict_temp
        mask_per_pred_new[modality] = dict_temp_pred

        mean_mask_new[modality] = np.mean(masks_transformed_new, axis=0)
        diff_mean_mask_new[modality] = np.mean(masks_transformed_new, axis=0) - np.mean(masks_transformed_comp, axis=0)

    # the mean and difference mask for this modality per target
    for modality, target_dict in masks_per_target_new.items():
        if modality not in mean_mask_new_target:
            mean_mask_new_target[modality] = {}
            diff_mean_mask_new_target[modality] = {}
            
        for target, masks_list in target_dict.items():
            mean_mask_new_target[modality][target] = np.mean(masks_list, axis=0)
            diff_mean_mask_new_target[modality][target] = (
                np.mean(masks_list, axis=0)
                - np.mean(masks_per_target_comp[modality][target], axis=0)
            )
    for modality, target_dict in mask_per_pred_new.items():
        if modality not in mean_mask_new_pred:
            mean_mask_new_pred[modality] = {}
            
        for target, masks_list in target_dict.items():
            mean_mask_new_pred[modality][target] = np.mean(masks_list, axis=0)
            

        # Add overall mean for this modality
        mean_mask_new_target[modality]["all"] = mean_mask_new[modality]
        diff_mean_mask_new_target[modality]["all"] = diff_mean_mask_new[modality]
        mean_mask_new_pred[modality]["all"] = mean_mask_new[modality]

    average_metrics = {key: np.mean(values) for key, values in metrics_all.items()}

        
    return average_metrics,mean_mask_new_target,diff_mean_mask_new_target,mean_mask_new_pred
    
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from typing import Dict, Any


def plot_masks(mask_dict, description, save_directory, title=None, ylabel ='Target'):
    """
    Plots masks for different modalities in a specified order and saves the plot and mask data.

    Args:
        mask_dict (dict): Dictionary where keys are modality identifiers and values are NumPy arrays of masks.
        description (str): Description used for naming the saved files.
        save_directory (str): Directory where the plot and data will be saved.
        title (str, optional): Title for the plot. Defaults to None.
    """
    import os
    import matplotlib.pyplot as plt
    import numpy as np
    import pickle
    import seaborn as sns

    try:
        # Create the save directory if it doesn't exist
        os.makedirs(save_directory, exist_ok=True)

        # Define new ordering of keys
        ordered_keys = [-3, -2, -1, 0, 1, 2, 3, 'all']
        ordered_labels = ["neg++", "neg+", "neg", "neu", "pos", "pos+", "pos++", "all"]
        
        # Sort and filter only present keys
        sorted_keys = [key for key in ordered_keys if key in mask_dict]
        sorted_labels = [ordered_labels[ordered_keys.index(key)] for key in sorted_keys]

        # Reorder the masks based on sorted keys
        sorted_masks = [mask_dict[key] for key in sorted_keys]

        # Verify that all masks have the same length
        mask_lengths = [mask.shape[0] for mask in sorted_masks]
        if len(set(mask_lengths)) != 1:
            raise ValueError("All masks must have the same length for plotting.")

        max_features = mask_lengths[0]  # Since all are the same

        # Convert masks to a 2D NumPy array
        masks_array = np.array(sorted_masks)

        # Check for NaN or Inf values
        if np.isnan(masks_array).any() or np.isinf(masks_array).any():
            print("Warning: Masks contain NaN or Inf values.")

        # Set up the figure
        plt.figure(figsize=(12, 4))

        # Use a perceptually uniform colormap like the second image
        sns.heatmap(masks_array, cmap='magma', xticklabels=5, yticklabels=sorted_labels, cbar=True)

        # Set axis labels
        plt.xlabel('Feature Dimension')
        plt.ylabel(ylabel)

        # Set the title
        plot_title = title if title else description.replace('_', ' ').capitalize()
        plt.title(plot_title)

        # Adjust layout
        plt.tight_layout()

        # Save the plot as an image **before** showing it
        plot_path = os.path.join(save_directory + "/plot_images", f"{description}.png")
        plt.savefig(plot_path)
        print(f"Plot saved to {plot_path}")

        # Show the plot
        plt.show()

        # Save the mask dictionary using pickle
        save_path = os.path.join(save_directory+ "/plot_numbers", f"{description}.pkl")
        with open(save_path, 'wb') as f:
            pickle.dump(mask_dict, f)
        print(f"Mask data saved to {save_path}")

    except Exception as e:
        print(f"An error occurred in plot_masks: {e}")




import os
import matplotlib.pyplot as plt
import numpy as np
import pickle
import seaborn as sns

import os
import matplotlib.pyplot as plt
import numpy as np
import pickle
import seaborn as sns

def prediction_count(data_new, data_comparison_link):
    data = load_comparison_data_pickle(data_comparison_link)
    predictions_new = data_new['predictions']
    predictions_comparison = data['predictions']
    targets = data_new['targets']
    predictions_distr_new = {}
    predictions_distr_comparison = {}
    targets_distr = {}
    
    label_mapping = {
        -3: "neg++",
        -2: "neg+",
        -1: "neg",
         0: "neu",
         1: "pos",
         2: "pos+",
         3: "pos++"
    }
    
    ordered_labels = [label_mapping[i] for i in sorted(label_mapping.keys())]
    
    for label in ordered_labels:
        predictions_distr_new[label] = 0
        predictions_distr_comparison[label] = 0
        targets_distr[label] = 0

    for i in range(len(predictions_new)):
        targets_distr[label_mapping[process_label(targets[i])]] += 1
        predictions_distr_new[label_mapping[process_label(predictions_new[i])]] += 1
        predictions_distr_comparison[label_mapping[process_label(predictions_comparison[i])]] += 1
    
    return predictions_distr_new, predictions_distr_comparison, targets_distr

def save_histogram_data(predictions_distr_new, predictions_distr_comparison, targets_distr, save_directory, experiment_name):
    os.makedirs(save_directory, exist_ok=True)
    
    ordered_labels = ["neg++", "neg+", "neg", "neu", "pos", "pos+", "pos++"]
    
    values_new = [predictions_distr_new[label] for label in ordered_labels]
    values_comparison = [predictions_distr_comparison[label] for label in ordered_labels]
    values_targets = [targets_distr[label] for label in ordered_labels]
    
    x = np.arange(len(ordered_labels))
    width = 0.3
    
    plt.figure(figsize=(10, 6))
    cmap = plt.get_cmap('magma')
    
    plt.bar(x - width, values_new, width, label='New Predictions', color=cmap(0.2), edgecolor='black')
    plt.bar(x, values_comparison, width, label='Base Model Predictions', color=cmap(0.5), edgecolor='black')
    plt.bar(x + width, values_targets, width, label='Targets', color=cmap(0.8), edgecolor='black')
    
    plt.xticks(x, ordered_labels, rotation=45)
    plt.xlabel('Prediction/Target Class')
    plt.ylabel('Count')
    plt.title('Prediction and Target Distributions')
    plt.legend()
    plt.tight_layout()
    
    # Save the figure
    plot_path = os.path.join(save_directory +"/plot_images", f"histogram.png")
    plt.savefig(plot_path)
    print(f"Histogram plot saved to {plot_path}")
    
    plt.show()
    
    distribution_data = {
        'predictions_distr_new': predictions_distr_new,
        'predictions_distr_comparison': predictions_distr_comparison,
        'targets_distr': targets_distr
    }
    save_path = os.path.join(save_directory+ "/plot_numbers", f"histogram.pkl") #
    with open(save_path, 'wb') as f:
        pickle.dump(distribution_data, f)
    print(f"Histogram distribution data saved to {save_path}")




def multiclass_acc(preds, truths):
    """
    Compute the multiclass accuracy w.r.t. groundtruth
    :param preds: Float array representing the predictions, dimension (N,)
    :param truths: Float/int array representing the groundtruth classes, dimension (N,)
    :return: Classification accuracy
    """
    return np.sum(np.round(preds) == np.round(truths)) / float(len(truths))


def weighted_accuracy(test_preds_emo, test_truth_emo):
    true_label = test_truth_emo > 0
    predicted_label = test_preds_emo > 0
    tp = float(np.sum((true_label == 1) & (predicted_label == 1)))
    tn = float(np.sum((true_label == 0) & (predicted_label == 0)))
    p = float(np.sum(true_label == 1))
    n = float(np.sum(true_label == 0))

    return (tp * (n / p) + tn) / (2 * n)


def eval_mosei_senti(results, truths, exclude_zero=False):
    test_preds = results.view(-1).cpu().detach().numpy()
    test_truth = truths.view(-1).cpu().detach().numpy()

    test_preds_a7 = np.clip(test_preds, a_min=-3.0, a_max=3.0)
    test_truth_a7 = np.clip(test_truth, a_min=-3.0, a_max=3.0)
    test_preds_a5 = np.clip(test_preds, a_min=-2.0, a_max=2.0)
    test_truth_a5 = np.clip(test_truth, a_min=-2.0, a_max=2.0)

    mae = np.mean(
        np.absolute(test_preds - test_truth)
    )  # Average L1 distance between preds and truths
    corr = np.corrcoef(test_preds, test_truth)[0][1]
    mult_a7 = multiclass_acc(test_preds_a7, test_truth_a7)
    mult_a5 = multiclass_acc(test_preds_a5, test_truth_a5)

    f_score = f1_score((test_preds >= 0), (test_truth >= 0), average="weighted")
    binary_truth = test_truth >= 0
    binary_preds = test_preds >= 0

    f_score_neg = f1_score((test_preds > 0), (test_truth > 0), average="weighted")
    binary_truth_neg = test_truth > 0
    binary_preds_neg = test_preds > 0

    non_zeros = np.array([i for i, e in enumerate(test_truth) if e != 0])
    f_score_non_zero = f1_score(
        (test_preds[non_zeros] > 0), (test_truth[non_zeros] > 0), average="weighted"
    )
    binary_truth_non_zero = test_truth[non_zeros] > 0
    binary_preds_non_zero = test_preds[non_zeros] > 0

    return {
        "mae": mae,
        "corr": corr,
        "acc_7": mult_a7,
        "acc_5": mult_a5,
        "f1_pos": f_score,  # zeros are positive
        "bin_acc_pos": accuracy_score(binary_truth, binary_preds),  # zeros are positive
        "f1_neg": f_score_neg,  # zeros are negative
        "bin_acc_neg": accuracy_score(
            binary_truth_neg, binary_preds_neg
        ),  # zeros are negative
        "f1": f_score_non_zero,  # zeros are excluded
        "bin_acc": accuracy_score(
            binary_truth_non_zero, binary_preds_non_zero
        ),  # zeros are excluded
    }


def eval_mosei_senti_old(results, truths, exclude_zero=False):
    test_preds = results.view(-1).cpu().detach().numpy()
    test_truth = truths.view(-1).cpu().detach().numpy()

    non_zeros = np.array(
        [i for i, e in enumerate(test_truth) if e != 0 or (not exclude_zero)]
    )

    test_preds_a7 = np.clip(test_preds, a_min=-3.0, a_max=3.0)
    test_truth_a7 = np.clip(test_truth, a_min=-3.0, a_max=3.0)
    test_preds_a5 = np.clip(test_preds, a_min=-2.0, a_max=2.0)
    test_truth_a5 = np.clip(test_truth, a_min=-2.0, a_max=2.0)

    mae = np.mean(
        np.absolute(test_preds - test_truth)
    )  # Average L1 distance between preds and truths
    corr = np.corrcoef(test_preds, test_truth)[0][1]
    mult_a7 = multiclass_acc(test_preds_a7, test_truth_a7)
    mult_a5 = multiclass_acc(test_preds_a5, test_truth_a5)
    f_score = f1_score(
        (test_preds[non_zeros] >= 0), (test_truth[non_zeros] >= 0), average="weighted"
    )
    binary_truth = test_truth[non_zeros] >= 0
    binary_preds = test_preds[non_zeros] >= 0
    f_score_neg = f1_score(
        (test_preds[non_zeros] > 0), (test_truth[non_zeros] > 0), average="weighted"
    )
    binary_truth_neg = test_truth[non_zeros] > 0
    binary_preds_neg = test_preds[non_zeros] > 0

    return {
        "mae": mae,
        "corr": corr,
        "acc_7": mult_a7,
        "acc_5": mult_a5,
        "f1": f_score,
        "f1_neg": f_score_neg,
        "bin_acc_neg": accuracy_score(binary_truth_neg, binary_preds_neg),
    }


def print_metrics(metrics):
    for k, v in metrics.items():
        print("{}:\t{}".format(k, v))


def save_metrics(metrics, results_file):
    with open(results_file, "w") as fd:
        for k, v in metrics.items():
            print("{}:\t{}".format(k, v), file=fd)


def eval_mosi(results, truths, exclude_zero=False):
    return eval_mosei_senti(results, truths, exclude_zero)


def eval_iemocap(results, truths, single=-1):
    emos = ["neutral", "happy", "sad", "angry"]
    test_preds = results.view(-1, 4, 2).cpu().detach().numpy()
    test_truth = truths.view(-1, 4).cpu().detach().numpy()

    results = {}
    for emo_ind in range(4):
        test_preds_i = np.argmax(test_preds[:, emo_ind], axis=1)
        test_truth_i = test_truth[:, emo_ind]
        f1 = f1_score(test_truth_i, test_preds_i, average="weighted")
        acc = accuracy_score(test_truth_i, test_preds_i)
        results["{}_acc".format(emos[emo_ind])] = acc
        results["{}_f1".format(emos[emo_ind])] = f1

    return results
