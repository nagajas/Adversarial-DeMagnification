import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.models import ResNet50_Weights
from argparse import ArgumentParser
from tqdm import tqdm
from collections import Counter
import numpy as np
import random
from PIL import Image

from train_defense import RLAgent, DefenseAgent, load_dataset
from deepfakes_dataset import DeepFakesDataset

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    parser = ArgumentParser()
    parser.add_argument('--image_path', type=str, required=True, help='Path to the input image')
    parser.add_argument('--classifier_path', type=str, default='../models_05_04/model_checkpoint20', help='Path to the trained classifier model')
    parser.add_argument('--model_path', type=str, default='defense_agent.pth', help='Path to the trained RL agent model')
    args = parser.parse_args()

    # Load classifier
    classifier = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    classifier.fc = nn.Linear(2048, 1)  # Binary classification
    classifier.load_state_dict(torch.load(args.classifier_path, map_location=device))
    classifier.to(device)
    classifier.eval()

    # Initialize the RL environment
    env = DefenseAgent(classifier, device=device)

    # Load the trained RL agent
    input_dim = 3 * 224 * 224  # Flattened image dimensions
    rl_agent = RLAgent(input_dim, env.action_space).to(device)
    rl_agent.load_state_dict(torch.load(args.model_path, map_location=device))
    rl_agent.eval()

    # Infer the label based on image filename if possible (optional)
    label = 0
    if os.path.basename(args.image_path).lower().startswith('fake'):
        label = 1
    elif os.path.basename(args.image_path).lower().startswith('real'):
        label = 0
    else:
        print("Warning: Couldn't infer label from filename. Assuming label=0 (REAL)")

    # Load and preprocess the image
    test_paths = [args.image_path]
    test_labels = [label]
    test_dataset = DeepFakesDataset(test_paths, test_labels)

    print("Running inference on the image...")

    #print(test_dataset[0])
    image, label,_ = test_dataset[0]

    with torch.no_grad():
        # Reset environment
        state = env.reset(image, label)
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state)
        state_tensor = state.to(device)

        # RL agent chooses an action
        q_values = rl_agent(state_tensor)
        action = q_values.argmax().item()

        # Take environment step
        obs, reward, info = env.step(action)

        # Permute image if needed
        if image.shape[0] == 224 and image.shape[2] == 3:
            image = image.permute(2, 0, 1)  # (H, W, C) -> (C, H, W)

        # Classifier prediction
        pred_logits = classifier(image.unsqueeze(0).to(device))
        prob = torch.sigmoid(pred_logits).item()

    print("\n--- Inference Result ---")
    print(f"Image Path: {args.image_path}")
    print(f"Predicted Probability of Being REAL: {100*prob:.2f}%")

    if prob < 0.5:
        print(f"Prediction: FAKE")
    else:
        print(f"Prediction: REAL")
    print("-------------------------\n")
