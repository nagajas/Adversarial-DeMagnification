import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.models import ResNet50_Weights
from argparse import ArgumentParser
from collections import Counter
from timm.scheduler.cosine_lr import CosineLRScheduler
import numpy as np
import random
from tqdm import tqdm

from deepfakes_dataset import DeepFakesDataset

class RLAgent(nn.Module):
    def __init__(self, input_dim, action_space):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, action_space)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        return self.fc2(x)
    
class DefenseAgent:
    def __init__(self, classifier, device='cuda'):
        self.classifier = classifier.to(device)
        self.device = device
        self.action_space = 2  # 0: do nothing, 1: mask center
        self.image_size = 224

    def reset(self, image, label):
        self.image = image.to(self.device)
        self.label = label.float().unsqueeze(0).to(self.device)
        return self.image.flatten()

    def step(self, action):
        image = self.image.clone()
        if action == 1:
            c = self.image_size // 2
            image[:, c-32:c+32, c-32:c+32] = 0  # -->masking
        
        if image.dim() == 3:
            image = image.unsqueeze(0) 
        
        #-->correct shape (B, C, H, W)
        image = image.permute(0, 3, 1, 2)  # [B, H, W, C] --> [B, C, H, W]

        with torch.no_grad():
            output = self.classifier(image)  # Classifier expects [B, C, H, W]
            prob = torch.sigmoid(output)
            pred = (prob > 0.5).float()
            correct = (pred == self.label).float().item()
        
        reward = +1 if correct else -1
        return image.flatten(), reward, True  # next_state, reward, done



def load_dataset(list_file):
    with open(list_file, 'r') as f:
        data = dict(json.load(f))

    annotations_json = data["annotations"]
    images_json = data['images']
    images_paths = []
    labels = []

    for item in annotations_json:
        for img_data in images_json:
            if img_data['id'] == item["image_id"]:
                item_path = img_data['file_name']
        label = item["category_id"]
        ip = os.path.join('..',item_path)
        if os.path.exists(ip):
            images_paths.append(ip)
            labels.append(label)
    print(f"Loaded {len(images_paths)} images from {list_file}")
    print(f"Labels: {Counter(labels)}")
    return images_paths, labels

def train_rl_agent(agent, env, optimizer, num_episodes, gamma=0.99, epsilon_start=1.0, 
                  epsilon_end=0.01, epsilon_decay=0.995, device='cuda'):
    """Train the RL agent to take optimal defensive actions"""
    epsilon = epsilon_start
    rewards_history = []
    
    # Ensure the agent is on the correct device
    agent.to(device)

    for episode in tqdm(range(num_episodes)):
        # Sample random image
        idx = random.randint(0, len(train_dataset) - 1)
        data = train_dataset[idx]
        image = data[0]  
        label = data[1]  
        
        # Reset environment
        state = env.reset(image, label)
        done = False
        episode_reward = 0
        
        # Convert state to tensor and move to the device
        if isinstance(state, torch.Tensor):
            state_tensor = state.to(device)
        else:
            state_tensor = torch.FloatTensor(state).to(device)
        
        # Epsilon-greedy action selection
        if random.random() < epsilon:
            action = random.randint(0, env.action_space - 1)
        else:
            with torch.no_grad():
                q_values = agent(state_tensor)
                action = q_values.argmax().item()
        
        # Take action and observe next state and reward
        next_state, reward, done = env.step(action)
        episode_reward += reward
        
        # Store transition and perform optimization
        optimizer.zero_grad()
        
        # Convert next state to tensor and move to device
        if isinstance(next_state, torch.Tensor):
            next_state_tensor = next_state.to(device)
        else:
            next_state_tensor = torch.FloatTensor(next_state).to(device)
        
        # Calculate Q-values
        q_values = agent(state_tensor)
        target_q_values = q_values.clone()
        
        # Update target for chosen action
        target_q_values[action] = reward
        
        # Compute loss and update
        loss = F.mse_loss(q_values, target_q_values)
        loss.backward()
        optimizer.step()
        
        # Update epsilon
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        
        rewards_history.append(episode_reward)
        
        # Print progress
        if (episode + 1) % 100 == 0:
            avg_reward = sum(rewards_history[-100:]) / 100
            print(f"Episode: {episode+1}, Avg Reward: {avg_reward:.2f}, Epsilon: {epsilon:.2f}")
    
    return rewards_history



if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    
    parser = ArgumentParser()
    parser.add_argument('--num_episodes', type=int, default=5000, help='Number of episodes for RL training')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--save_path', type=str, default='./defense_agent.pth', help='Path to save model')
    args = parser.parse_args()
    
    # Load datasets
    train_paths, train_labels = load_dataset('../Images/train_data.json')
    val_paths, val_labels = load_dataset('../Images/val_data.json')

    train_dataset = DeepFakesDataset(train_paths, train_labels)
    train_dl = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, prefetch_factor=2)
    
    val_dataset = DeepFakesDataset(val_paths, val_labels)
    val_dl = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, prefetch_factor=2)
    # for i, data in enumerate(tqdm(val_dl)):
    #     print(f"Batch {i}:")
    #     print(f"Data type: {type(data)}")
    #     print(f"Data content: {data}")  # To see the full content of one batch
    #     break
        
    # Print dataset statistics
    train_samples = len(train_dataset)
    val_samples = len(val_dataset)

    print(f"Train samples: {train_samples}")
    print(f"Validation samples: {val_samples}")

    print(f'TRAINING STATS')
    train_counters = dict(Counter(train_labels))
    print(f'Train dataset distribution: {train_counters}')
    
    cls_weights = train_counters[0] / train_counters[1]
    print('Weights', cls_weights)

    print(f'VALIDATION STATS')
    val_counters = dict(Counter(val_labels))
    print(f'Validation dataset distribution: {val_counters}')
    print('Weights', val_counters[0] / val_counters[1])
    
    classifier = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    classifier.fc = nn.Linear(2048, 1)  # Binary classification
    model_path = '../models_05_04/model_checkpoint20'
    classifier.load_state_dict(torch.load(model_path, map_location=device))
    classifier.to(device)
    classifier.eval()

    print("Evaluating pre-trained classifier on validation set (baseline)...")
    base_tp, base_fp, base_tn, base_fn = 0, 0, 0, 0

    with torch.no_grad():
        for i, d in enumerate(tqdm(val_dl)):
            if i >= 100:
                break
                
            images, labels = d[0].to(device), d[1].to(device)

            if images.dim() == 4 and images.shape[3] == 3:
                 images = images.permute(0, 3, 1, 2)
            
            outputs = classifier(images)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            
            for pred, label in zip(preds, labels):
                if pred.item() == 1 and label.item() == 1:  # True Positive
                    base_tp += 1
                elif pred.item() == 1 and label.item() == 0:  # False Positive
                    base_fp += 1
                elif pred.item() == 0 and label.item() == 0:  # True Negative
                    base_tn += 1
                elif pred.item() == 0 and label.item() == 1:  # False Negative
                    base_fn += 1

    base_tpr = base_tp / (base_tp + base_fn) if (base_tp + base_fn) > 0 else 0  # True Positive Rate
    base_fpr = base_fp / (base_fp + base_tn) if (base_fp + base_tn) > 0 else 0  # False Positive Rate
    base_tnr = base_tn / (base_tn + base_fp) if (base_tn + base_fp) > 0 else 0  # True Negative Rate (Specificity)
    base_fnr = base_fn / (base_fn + base_tp) if (base_fn + base_tp) > 0 else 0  # False Negative Rate (Miss Rate)
    base_acc = (base_tp + base_tn) / (base_tp + base_tn + base_fp + base_fn)   # Accuracy

    print(f"Baseline Classifier Results:")
    print(f"TPR (True Positive Rate/Recall): {base_tpr:.4f}")
    print(f"FPR (False Positive Rate): {base_fpr:.4f}")
    print(f"TNR (True Negative Rate/Specificity): {base_tnr:.4f}")
    print(f"FNR (False Negative Rate): {base_fnr:.4f}")
    print(f"Accuracy: {base_acc:.4f}")
    print("-" * 50)

    
    # Initialize the RL environment
    env = DefenseAgent(classifier, device=device)
    
    # Initialize the RL agent
    input_dim = 3 * 224 * 224  # Flattened image dimensions
    rl_agent = RLAgent(input_dim, env.action_space).to(device)
    optimizer = optim.Adam(rl_agent.parameters(), lr=args.lr)
    
    # Train the RL agent
    print(f"Starting RL training for {args.num_episodes} episodes...")
    rewards = train_rl_agent(rl_agent, env, optimizer, args.num_episodes, device=device)
    
    # Save the trained agent
    torch.save(rl_agent.state_dict(), args.save_path)
    print(f"RL agent saved to {args.save_path}")
    
    # Evaluate the RL agent on validation set
    print("Evaluating RL agent on validation set...")
    tp, fp, tn, fn = 0, 0, 0, 0  # Initialize counts for TP, FP, TN, FN

    for i, d in enumerate(tqdm(val_dl)):
        if i >= 100:  # Limit evaluation for speed
            break

        images, labels = d[0], d[1]

        for image, label in zip(images, labels):
            # Reset environment with this image
            state = env.reset(image, label)

            # Ensure state is a tensor and move it to the correct device
            if not isinstance(state, torch.Tensor):
                state = torch.FloatTensor(state)
            state_tensor = state.to(device)

            # Select action using trained agent
            with torch.no_grad():
                q_values = rl_agent(state_tensor)
                action = q_values.argmax().item()

            # Apply action and check if correctly classified
            _, reward, _ = env.step(action)

            # Update TP, FP, TN, FN counts
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

    # Print results
    print(f"Validation Results:")
    print(f"TPR (True Positive Rate): {tpr}")
    print(f"FPR (False Positive Rate): {fpr}")
    print(f"TNR (True Negative Rate): {tnr}")
    print(f"FNR (False Negative Rate): {fnr}")