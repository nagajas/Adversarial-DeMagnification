import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models
from argparse import ArgumentParser
from tqdm import tqdm
from collections import Counter
from torchvision.models import ResNet50_Weights
import numpy as np
import random

from train_defense import RLAgent, DefenseAgent, load_dataset
from deepfakes_dataset import DeepFakesDataset

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    parser = ArgumentParser()
    parser.add_argument('--test_data', type=str, default='../Images/test_data.json', help='Path to test dataset JSON file')
    parser.add_argument('--model_path', type=str, default='defense_agent.pth', help='Path to the trained RL agent model')
    args = parser.parse_args()

    # Load test dataset
    test_paths, test_labels = load_dataset(args.test_data)
    test_dataset = DeepFakesDataset(test_paths, test_labels)
    test_dl = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, prefetch_factor=2)

    classifier = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    classifier.fc = nn.Linear(2048, 1)  # Binary classification
    model_path = '../models_05_04/model_checkpoint20'
    classifier.load_state_dict(torch.load(model_path, map_location=device))
    classifier.to(device)
    classifier.eval()

    # Initialize the RL environment
    env = DefenseAgent(classifier, device=device)

    # Load the trained RL agent
    input_dim = 3 * 224 * 224  # Flattened image dimensions
    rl_agent = RLAgent(input_dim, env.action_space).to(device)
    rl_agent.load_state_dict(torch.load(args.model_path, map_location=device))
    rl_agent.eval()

    print("Evaluating RL agent on test set...")
    tp, fp, tn, fn = 0, 0, 0, 0  # Initialize counts for TP, FP, TN, FN

    for i, d in enumerate(tqdm(test_dl)):
        images, labels = d[0], d[1]

        for image, label in zip(images, labels):
            # Reset environment
            state = env.reset(image, label)

            if not isinstance(state, torch.Tensor):
                state = torch.FloatTensor(state)
            state_tensor = state.to(device)

            with torch.no_grad():
                q_values = rl_agent(state_tensor)
                action = q_values.argmax().item()

            _, reward, _ = env.step(action)

            if reward > 0:  # Correct classification
                if label == 1:  # True Positive
                    tp += 1
                else:  # True Negative
                    tn += 1
            else:  # Incorrect classification
                if label == 1:  # False Negative
                    fn += 1
                else:  # False Positive
                    fp += 1

    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0  # True Positive Rate
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0  # True Negative Rate
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0  # False Negative Rate

    print(f"Test Results:")
    print(f"TPR (True Positive Rate): {tpr:.2f}")
    print(f"FPR (False Positive Rate): {fpr:.2f}")
    print(f"TNR (True Negative Rate): {tnr:.2f}")
    print(f"FNR (False Negative Rate): {fnr:.2f}")