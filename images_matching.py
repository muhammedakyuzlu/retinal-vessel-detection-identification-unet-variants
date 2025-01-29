import glob
import logging
import math
import os
import pickle
import json
from datetime import datetime
from functools import wraps
from typing import Dict, Tuple, Any

import cv2
import numpy as np
import torch


def timeit(func):
    """Decorator to measure the execution time of functions."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = datetime.now()
        result = func(*args, **kwargs)
        end_time = datetime.now()
        logging.info(f"Function '{func.__name__}' executed in {end_time - start_time}")
        return result
    return wrapper


def save_pickle(data: Any, path: str):
    """Save data to a pickle file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f)


def load_pickle(path: str) -> Any:
    """Load data from a pickle file."""
    with open(path, "rb") as f:
        return pickle.load(f)


def save_json(data: Any, path: str):
    """Save data to a JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)


def load_json(path: str) -> Any:
    """Load data from a JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def _convert_text(input_text: str) -> str:
    """
    Convert input text to a standardized format.

    Parameters:
        input_text (str): The input text to convert.

    Returns:
        str: The converted text.
    """
    parts = input_text.split("_")
    if len(parts) < 3:
        logging.warning(f"Unexpected format in input_text: {input_text}")
        return input_text
    direction = parts[2].split("-")[0]
    return f"{parts[0]}{direction}"


def create_descriptor(
    root_dir: str, descriptor_method: Tuple[str, dict], save_path: str
):
    """
    Create and save image descriptors using the specified method.

    Parameters:
        root_dir (str): Directory containing images.
        descriptor_method (Tuple[str, dict]): Tuple containing the method name and its parameters.
        save_path (str): Path to save the descriptors.
    """
    method_name, params = descriptor_method
    image_descriptors = {}

    # Initialize the feature detector based on the method
    if method_name == "sift":
        feature_detector = cv2.SIFT_create(**params)
    elif method_name == "surf":
        feature_detector = cv2.xfeatures2d.SURF_create(**params)
    elif method_name == "brisk":
        feature_detector = cv2.BRISK_create(**params)
    elif method_name == "fast":
        feature_detector = cv2.FastFeatureDetector_create(
            nonmaxSuppression=params.get("NMS", True)
        )
    elif method_name == "harris":
        pass  # Harris will be handled separately
    else:
        raise ValueError(f"Unsupported descriptor method: {method_name}")

    big_files = []
    image_paths = glob.glob(os.path.join(root_dir, "*"))
    for count, path in enumerate(image_paths, 1):
        image = cv2.imread(path)
        name = os.path.splitext(os.path.basename(path))[0]

        if method_name == "fast":
            keypoints = feature_detector.detect(image, None)
            brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
            keypoints, descriptors = brief.compute(image, keypoints)
        elif method_name == "harris":
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            keypoints = cv2.goodFeaturesToTrack(gray_image, **params)
            if keypoints is None:
                logging.warning(f"No keypoints found for image: {name}")
                continue
            keypoints = [
                cv2.KeyPoint(x=float(pt[0][0]), y=float(pt[0][1]), size=1)
                for pt in keypoints
            ]
            brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
            keypoints, descriptors = brief.compute(gray_image, keypoints)
        else:
            keypoints, descriptors = feature_detector.detectAndCompute(image, None)

        if descriptors is None:
            logging.warning(f"No descriptors found for image: {name}")
            continue

        image_descriptors[name] = descriptors

        if len(descriptors) > 15000:
            big_files.append((name, len(descriptors)))

        logging.info(f"Processed image {count}/{len(image_paths)}: {name}")

    if big_files:
        logging.info(f"Images with large descriptor counts: {big_files}")

    save_pickle(image_descriptors, save_path)
    logging.info(f"Descriptors saved to {save_path}")


def calculate_individual_accuracy(data: Dict[str, Tuple[str, float]]) -> float:
    """
    Calculate the accuracy based on matching results.

    Parameters:
        data (dict): Dictionary containing match results.

    Returns:
        float: The calculated accuracy.
    """
    correct_matches = 0
    total = len(data)
    for key, value in data.items():
        k_name = _convert_text(key)
        v_name = _convert_text(value[0])
        if k_name == v_name:
            correct_matches += 1
    accuracy = correct_matches / total if total > 0 else 0
    return accuracy


def calculate_accuracy(
    match_percentages: Dict[str, Tuple[int, float]]
) -> Dict[str, Any]:
    """
    Calculate the overall accuracy from match percentages.

    Parameters:
        match_percentages (dict): Dictionary containing match percentages.

    Returns:
        dict: A dictionary containing accuracy data.
    """
    score_data = {}
    feature_data = {}

    for key, value in match_percentages.items():
        first_image, second_image = key.split("=")
        if first_image == second_image:
            continue
        features, score = value
        if first_image not in score_data or score_data[first_image][1] < score:
            score_data[first_image] = (second_image, score)
        if first_image not in feature_data or feature_data[first_image][1] < features:
            feature_data[first_image] = (second_image, features)

    score_accuracy = calculate_individual_accuracy(score_data)
    feature_accuracy = calculate_individual_accuracy(feature_data)

    accuracy = {
        "score_data": score_data,
        "feature_data": feature_data,
        "score_accuracy": score_accuracy,
        "feature_accuracy": feature_accuracy,
    }

    return accuracy


def cpu_compare_descriptors(
    des1: np.ndarray, des2: np.ndarray
) -> Tuple[int, float]:
    """
    Compare two descriptors using CPU.

    Parameters:
        des1 (np.ndarray): First descriptor.
        des2 (np.ndarray): Second descriptor.

    Returns:
        Tuple[int, float]: Number of matches and match percentage.
    """
    if des1 is None or des2 is None or len(des1) == 0 or len(des2) == 0:
        return 0, 0.0

    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = matcher.match(des1, des2)
    match_percentage = math.sqrt((len(matches) ** 2) / (len(des1) * len(des2))) * 100

    return len(matches), match_percentage


def gpu_compare_descriptors(
    des1: torch.Tensor, des2: torch.Tensor
) -> Tuple[int, float]:
    """
    Compare two descriptors using GPU.

    Parameters:
        des1 (torch.Tensor): First descriptor tensor.
        des2 (torch.Tensor): Second descriptor tensor.

    Returns:
        Tuple[int, float]: Number of matches and match percentage.
    """
    des1 = des1.float()
    des2 = des2.float()

    if des1.nelement() == 0 or des2.nelement() == 0:
        return 0, 0.0

    distances = torch.cdist(des1, des2)
    min_dist_idx_des1 = torch.argmin(distances, dim=1)
    min_dist_idx_des2 = torch.argmin(distances, dim=0)

    mutual_matches = (
        min_dist_idx_des2[min_dist_idx_des1] == torch.arange(des1.size(0), device=des1.device)
    )
    match_count = mutual_matches.sum().item()
    match_percentage = math.sqrt((match_count ** 2) / (des1.size(0) * des2.size(0))) * 100

    return match_count, match_percentage


@timeit
def compare_descriptors(
    descriptors: Dict[str, Any], device: str = "gpu"
) -> Dict[str, Tuple[int, float]]:
    """
    Compare descriptors and compute match results.

    Parameters:
        descriptors (dict): Dictionary of descriptors.
        device (str): 'cpu' or 'gpu'.

    Returns:
        dict: Match results with match counts and percentages.
    """
    match_results = {}
    keys = list(descriptors.keys())
    n_keys = len(keys)

    for i in range(n_keys):
        for j in range(n_keys):
            key_i = keys[i]
            key_j = keys[j]
            compound_key = f"{key_i}={key_j}"
            if compound_key in match_results:
                continue

            des1 = descriptors[key_i]
            des2 = descriptors[key_j]

            if device == "gpu":
                features_nums, score = gpu_compare_descriptors(des1, des2)
            else:
                features_nums, score = cpu_compare_descriptors(des1, des2)

            match_results[compound_key] = (features_nums, score)

    return match_results

def calculate_confusion_matrix(match_percentages: Dict[str, Tuple[int, float]], threshold: float = 20) -> np.ndarray:
    """
    Calculate the confusion matrix based on match percentages and a threshold.

    Parameters:
        match_percentages (Dict[str, Tuple[int, float]]): Dictionary containing match percentages.
        threshold (float): Threshold for determining positive matches.

    Returns:
        np.ndarray: A 2x2 confusion matrix.
    """
    accuracy_labels = {}
    for key, value in match_percentages.items():
        first_image = key
        second_image = value[0]
        if first_image == second_image:
            continue
        if first_image in accuracy_labels:
            if accuracy_labels.get(first_image, [None, 0])[1] < value[1]:
                accuracy_labels[first_image] = [second_image, value[1]]
        else:
            accuracy_labels[first_image] = [second_image, value[1]]

    dict_sorted_keys = {k: accuracy_labels[k] for k in sorted(accuracy_labels)}

    confusion_matrix = np.zeros((2, 2), dtype=int)
    # Confusion matrix format:
    # [[TP, FP],
    #  [FN, TN]]

    for key, value in dict_sorted_keys.items():
        match_score = value[1]
        k_name = _convert_text(key)
        v_name = _convert_text(value[0])

        if k_name == v_name:
            if match_score > threshold:
                confusion_matrix[0][0] += 1  # TP
            else:
                confusion_matrix[1][0] += 1  # FN
        else:
            if match_score > threshold:
                confusion_matrix[0][1] += 1  # FP
            else:
                confusion_matrix[1][1] += 1  # TN

    return confusion_matrix

def calculate_performance_metrics(confusion_matrix: np.ndarray) -> Tuple[float, float, float, float, float]:
    """
    Calculate performance metrics based on the confusion matrix.

    Parameters:
        confusion_matrix (np.ndarray): A 2x2 confusion matrix.

    Returns:
        Tuple[float, float, float, float, float]: FAR, FRR, precision, recall, and accuracy.
    """
    TP, FP = confusion_matrix[0, :]
    FN, TN = confusion_matrix[1, :]

    # FAR: False Acceptance Rate
    FAR = FP / (FP + TN) if (FP + TN) > 0 else 0

    # FRR: False Rejection Rate
    FRR = FN / (FN + TP) if (FN + TP) > 0 else 0

    # Precision
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0

    # Recall
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0

    # Accuracy
    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0

    return FAR, FRR, precision, recall, accuracy

def get_int_statistics(data: Dict[str, Tuple[str, float]]) -> Tuple[float, float, float]:
    """
    Calculate maximum, minimum, and average of match scores.

    Parameters:
        data (Dict[str, Tuple[str, float]]): Dictionary containing match scores.

    Returns:
        Tuple[float, float, float]: Maximum, minimum, and average scores.
    """
    if not data:
        return None, None, None  # If the dictionary is empty

    float_values = [int(value[1]) for value in data.values()]
    total = sum(float_values)
    max_value = max(float_values)
    min_value = min(float_values)
    average = int(total / len(float_values))

    return max_value, min_value, average

def calculate_far_frr_metrics(
    accuracy: Dict[str, Dict[str, Tuple[str, float]]],
    save_path: str,
    method_name: str,
    fSave: Any
) -> None:
    """
    Compute the best threshold metrics from the accuracy data and save them to a JSON file.

    Parameters:
        accuracy (Dict[str, Dict[str, Tuple[str, float]]]): The accuracy data containing 'score_data' and 'feature_data'.
        save_path (str): Path to save results.
        method_name (str): The method name used in descriptors.
        fSave (Any): Additional identifier for saving files.

    Returns:
        None
    """
    metrics_results: Dict[str, Any] = {}
    
    
    data_types = {"score_data": accuracy["score_data"], "feature_data": accuracy["feature_data"]}


    for data_type, data in data_types.items():
        

        # Initialize variables to store the best FAR, FRR, and their corresponding threshold
        best_far = 0
        best_frr = 100
        best_threshold = None
        best_metrics = {}

        max_value, min_value, average = get_int_statistics(data)

        if data_type == "score_data":
            step = 0.1
        else:
            step = 1    

        for threshold in np.arange(min_value, max_value,step):    
            
            confusion_matrix = calculate_confusion_matrix(data, threshold)

            far,frr,precision,recall,accuracy = calculate_performance_metrics(confusion_matrix)
            
            metrics = {
                "Threshold": float(threshold),
                "FAR": far,
                "FRR": frr,
                "Precision": precision,
                "Recall": recall,
                "Accuracy": accuracy,
                "Confusion Matrix": confusion_matrix.tolist()
            }

            # Update the best values if the current FAR and FRR are better balanced
            if abs(far - frr) < abs(best_far - best_frr):
                best_far = far
                best_frr = frr
                best_threshold = threshold
                best_metrics = metrics
            
        metrics_results[data_type] = {
                    "max_value":max_value,
                    "min_value":min_value, 
                    "average":average,
                    "TP":int(best_metrics["Confusion Matrix"][0][0]),
                    "FP":int(best_metrics["Confusion Matrix"][0][1]),
                    "FN":int(best_metrics["Confusion Matrix"][1][0]),
                    "TN":int(best_metrics["Confusion Matrix"][1][1]),
                    "best_metrics":best_metrics,
                }

    metrics_file_path = save_path+f"/{method_name}/{method_name}_metrics_F{fSave}.json"
    print(metrics_file_path)
    save_json(metrics_results,metrics_file_path)





def process_and_match_descriptors(
    descriptor_method: Tuple[str, dict], base_path: str, save_path: str, device: str = "gpu"
):
    """
    Process images to create descriptors and match them.

    Parameters:
        descriptor_method (Tuple[str, dict]): Descriptor method and its parameters.
        base_path (str): Path to the images.
        save_path (str): Path to save results.
        device (str): 'cpu' or 'gpu' for computation.
    """
    method_name, params = descriptor_method
    fSave = params.pop('fSave', None)  # Extract and remove 'fSave'
    # descriptor_save_path = os.path.join(
    #     save_path,
    #     method_name,
    #     f"{method_name}_descriptor_F{fSave}.pkl"
    # )

    # create_descriptor(base_path, descriptor_method, descriptor_save_path)
    # descriptor_data = load_pickle(descriptor_save_path)

    # descriptors = {}
    # for key, desc in descriptor_data.items():
    #     if desc is None:
    #         logging.info(f"Skipping key {key} because it has no descriptors")
    #         continue
    #     if device == "gpu":
    #         descriptors[key] = torch.tensor(desc).cuda()
    #     else:
    #         descriptors[key] = desc

    # match_results = compare_descriptors(descriptors, device)

    # match_save_path = os.path.join(
    #     save_path,
    #     method_name,
    #     f"{method_name}_match_{device}_F{fSave}.json"
    # )
    # save_json(match_results, match_save_path)
    # logging.info(f"Match results saved to {match_save_path}")

    # match_percentages = load_json(match_save_path)
    # accuracy = calculate_accuracy(match_percentages)

    accuracy_save_path = os.path.join(
        save_path,
        method_name,
        f"{method_name}_accuracy_F{fSave}.json"
    )
    # save_json(accuracy, accuracy_save_path)
    # logging.info(f"Accuracy results saved to {accuracy_save_path}")

    accuracy = load_json(accuracy_save_path)
    calculate_far_frr_metrics(accuracy, save_path,method_name,fSave)

if __name__ == "__main__":


    start_time = datetime.now()

    ###########  config ############
    # Base directories

    # images directory
    images_path = "/workspace/unet_images"
    # results directory
    save_path = "/workspace/results"



    # harris
    harris_config = {
        "maxCorners": 1000,
        "qualityLevel": 0.01,
        "minDistance":10,
        "blockSize": 5,
        "useHarrisDetector": False,
        "k": 0.04,
        "mask": None,
        "fSave": 1000, # save file name
    }		
   
    # fast
    fast_config = {
        "NMS":False,
        "fSave": False # save file name
    }

    # brisk
    breaker_config = {
        "thresh":0,
        "octaves":3,
        "fSave":3_0, # save file name
    }
   
    # sift config
    sift_config = {
        "nfeatures": 0,  # no limit take all the features
        "nOctaveLayers": 30,
        "contrastThreshold": 0.04,
        "edgeThreshold": 1000,
        "sigma": 1.6,
        "fSave":0, # save file name
    }
    
    # surf config
    surf_config = {
        "hessianThreshold": 100,
        "nOctaves": 4,
        "nOctaveLayers": 30,
        "extended": False,
        "upright": False,
        "fSave": 4, # save file name
    }





    descriptor_method = ["fast", fast_config]
    # descriptor_method = ["harris", harris_config]
    # descriptor_method = ["brisk", breaker_config]
    # descriptor_method = ["sift", sift_config]
    # descriptor_method = ["surf", surf_config]



    process_and_match_descriptors(descriptor_method, images_path, save_path, device="gpu")

    end_time = datetime.now()
    logging.info(end_time - start_time)
