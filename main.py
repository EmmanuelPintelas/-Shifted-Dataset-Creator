
from pathlib import Path
import numpy as np
from scipy.optimize import dual_annealing
import os
import shutil
from glob import glob
from PIL import Image
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
import torch
import timm
import torchvision.transforms as transforms
import torch.nn.functional as F
from scipy.spatial.distance import jensenshannon
import matplotlib.pyplot as plt
import umap
from matplotlib.lines import Line2D
from matplotlib.cm import get_cmap
import random

from CONFIGS import *
ROOT_DIR = USER_CONFIG['ROOT_DIR']
OUTPUT_DIR = USER_CONFIG['OUTPUT_DIR']
TRAIN_RATIO = USER_CONFIG['SPLIT_RATIO']
SHIFT_MAGNITUDE = USER_CONFIG['SHIFT_MAGNITUDE']
VISUALIZER = USER_CONFIG['VISUALIZER']

# --- clean up ---
folder = USER_CONFIG["OUTPUT_DIR"]

try:
    for name in os.listdir(folder):
        path = os.path.join(folder, name)

        if os.path.isdir(path):
            shutil.rmtree(path)     # delete directory
        else:
            os.remove(path)         # delete file
except:
    print()
# --- clean up ---


def save_split(base_dir, split_name, paths, labels):
    for path, label in zip(paths, labels):
        dest_dir = os.path.join(base_dir, split_name, label)
        os.makedirs(dest_dir, exist_ok=True)
        shutil.copy(path, os.path.join(dest_dir, os.path.basename(path)))


def load_image_paths_labels(folder):
    paths, labels = [], []
    for label in sorted(os.listdir(folder)):
        label_dir = os.path.join(folder, label)
        if os.path.isdir(label_dir):
            image_paths = glob(os.path.join(label_dir, '*'))
            paths.extend(image_paths)
            labels.extend([label] * len(image_paths))
    return np.array(paths), np.array(labels)


def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])


class FeatureEncoder(torch.nn.Module):
    def __init__(self, model_name="mobilenetv3_small_100", device="cpu"):
        super().__init__()
        self.encoder = timm.create_model(model_name, pretrained=True, features_only=True)
        self.encoder.eval()
        self.device = torch.device(device)
        self.encoder.to(self.device)

    def forward(self, x):
        with torch.no_grad():
            features = self.encoder(x)[-1]
            pooled = F.adaptive_avg_pool2d(features, 1)
            return pooled.view(pooled.size(0), -1)

    def encode_batch(self, image_batch):
        image_batch = image_batch.to(self.device)
        return self.forward(image_batch)


def extract_softmax_distribution(paths, encoder, transform, batch_size=32):
    outputs = []
    device = encoder.device
    for i in tqdm(range(0, len(paths), batch_size), desc="Batch encoding"):
        batch_paths = paths[i:i + batch_size]
        images = []
        for path in batch_paths:
            img = Image.open(path).convert("RGB")
            images.append(transform(img))
        batch_tensor = torch.stack(images).to(device)
        embeddings = encoder.encode_batch(batch_tensor)
        softmaxed = torch.nn.functional.softmax(embeddings, dim=1)
        outputs.append(softmaxed.cpu().numpy())
    return np.vstack(outputs)


class JSDRatioConstrainedClustering:
    def __init__(self, target_ratio=TRAIN_RATIO, max_iter=50, tol=1e-4, seed=42):
        self.target_ratio = target_ratio
        self.max_iter = max_iter
        self.tol = tol
        self.seed = seed

    @staticmethod
    def compute_jsd(P, Q):
        return jensenshannon(P, Q, base=2)**2

    def cluster(self, X):
        np.random.seed(self.seed)
        N = X.shape[0]
        n_train = int(self.target_ratio * N)

        idx_train = np.random.choice(N, size=n_train, replace=False)
        idx_test = np.setdiff1d(np.arange(N), idx_train)

        prev_jsd = -1

        for _ in range(self.max_iter):
            mu_train = X[idx_train].mean(axis=0)
            mu_test = X[idx_test].mean(axis=0)

            print('JSD: ', self.compute_jsd(mu_train, mu_test))

            jsd_to_train = np.array([self.compute_jsd(x, mu_train) for x in X])
            jsd_to_test = np.array([self.compute_jsd(x, mu_test) for x in X])

            jsd_diff = jsd_to_test - jsd_to_train
            sorted_indices = np.argsort(jsd_diff)

            new_idx_train = sorted_indices[:n_train]
            new_idx_test = sorted_indices[n_train:]

            mu_new_train = X[new_idx_train].mean(axis=0)
            mu_new_test = X[new_idx_test].mean(axis=0)
            current_jsd = self.compute_jsd(mu_new_train, mu_new_test)


            if current_jsd - prev_jsd < self.tol:#current_jsd>0.4:#
                ##idx_train, idx_test = new_idx_train, new_idx_test
                break
            else:
                idx_train, idx_test = new_idx_train, new_idx_test
                prev_jsd = current_jsd

        final_jsd = self.compute_jsd(X[idx_train].mean(axis=0), X[idx_test].mean(axis=0))
        ##print('FINAL JSD: (per class)', self.compute_jsd(X[idx_train].mean(axis=0), X[idx_test].mean(axis=0)))
        return idx_train, idx_test, final_jsd
    


def visualize_split(embeddings, labels, split_indices, split_names, le, max_points_per_class=400):
    reducer = umap.UMAP(random_state=42)
    embeddings_2d = reducer.fit_transform(embeddings)

    plt.figure(figsize=(12, 12))

    # Define distinct markers for classes
    markers = ['o', 's', '^', 'D', 'v', 'P', '*', 'X']  # Extend if more classes
    unique_labels = np.unique(labels)
    marker_dict = {label: markers[i % len(markers)] for i, label in enumerate(unique_labels)}

    # Color map with enough separation
    cmap = get_cmap('tab10') if len(unique_labels) <= 10 else get_cmap('tab20')
    color_dict = {label: cmap(i % cmap.N) for i, label in enumerate(unique_labels)}

    for split_idx, split_name in zip(split_indices, split_names):
        split_embeds = embeddings_2d[split_idx]
        split_labels = labels[split_idx]

        for label in unique_labels:
            label_mask = split_labels == label
            indices = np.where(split_idx)[0][label_mask]
            
            # Subsample if too many points
            if len(indices) > max_points_per_class:
                indices = np.random.choice(indices, size=max_points_per_class, replace=False)

            emb_points = embeddings_2d[indices]
            marker = marker_dict[label]
            color = color_dict[label]
            label_name = le.inverse_transform([label])[0]

            # Train = unfilled, Test = filled
            if 'train' in split_name.lower():
                plt.scatter(
                    emb_points[:, 0], emb_points[:, 1],
                    label=f"{split_name.upper()} - {label_name}",
                    marker=marker,
                    edgecolors=color,
                    facecolors='none',
                    linewidths=1.0,
                    s=40
                )
            else:
                plt.scatter(
                    emb_points[:, 0], emb_points[:, 1],
                    label=f"{split_name.upper()} - {label_name}",
                    marker=marker,
                    color=color,
                    edgecolors='black',
                    linewidths=0.3,
                    alpha=0.8,
                    s=40
                )

    plt.title("Train/Test Distribution with UMAP Projection")
    plt.legend(fontsize=8, loc='best', ncol=2)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel("")
    plt.ylabel("")
    plt.grid(False)

    # Save to file
    plt.tight_layout()
    plt.savefig("umap_plot.png", dpi=500, bbox_inches='tight')
    plt.close()






device = "cuda" if torch.cuda.is_available() else "cpu"
encoder = FeatureEncoder(model_name="mobilenetv3_small_100", device=device)
transform = get_transform()


if os.path.exists(CACHE_PATH):
        print("\n✅ Loading cached embeddings...")
        cached = np.load(CACHE_PATH, allow_pickle=True)
        embeddings = cached['embeddings']
        labels = cached['labels']
        paths = cached['paths']
else:
        print("\n...Computing embeddings...")
        paths, labels = load_image_paths_labels(ROOT_DIR)
        embeddings = extract_softmax_distribution(paths, encoder, transform)
        np.savez(CACHE_PATH, embeddings=embeddings, labels=labels, paths=paths)
        print("\n✅ Embeddings cached!")

print("\n...The JSD Spliting Algorithm starts...")
le = LabelEncoder()
labels_encoded = le.fit_transform(labels)
train_paths, train_labels, test_paths, test_labels = [], [], [], []

total_jsd_score = 0
for cls in np.unique(labels_encoded):
        idx = np.where(labels_encoded == cls)[0]
        cls_embeddings = embeddings[idx]
        cls_paths = paths[idx]
        cls_labels = labels[idx]

        clusterer = JSDRatioConstrainedClustering(target_ratio=TRAIN_RATIO)
        idx_train, idx_test, jsd_score = clusterer.cluster(cls_embeddings)

        train_paths.extend(cls_paths[idx_train])
        train_labels.extend(cls_labels[idx_train])
        test_paths.extend(cls_paths[idx_test])
        test_labels.extend(cls_labels[idx_test])

        total_jsd_score += jsd_score


total_jsd_score = total_jsd_score / len(np.unique(labels_encoded))

print("\nThe JSD Spliting Algorithm has terminated ✅")
print(f"\nTOTAL JSD: {np.round(total_jsd_score, 5)}")



if SHIFT_MAGNITUDE > 0:

    zeta = min(max(float(SHIFT_MAGNITUDE), 0), 0.5)

    new_train_paths = []
    new_train_labels = []
    new_test_paths  = []
    new_test_labels = []

    test_paths  = np.array(test_paths)
    test_labels = np.array(test_labels)
    train_paths = np.array(train_paths)
    train_labels = np.array(train_labels)

    for cls in np.unique(train_labels):

        # ---- Gather per-class items from TRAIN ----
        train_idx = np.where(train_labels == cls)[0]
        paths_train_c  = train_paths[train_idx]
        labels_train_c = train_labels[train_idx]

        # ---- Gather per-class items from TEST ----
        test_idx = np.where(test_labels == cls)[0]
        paths_test_c  = test_paths[test_idx]
        labels_test_c = test_labels[test_idx]

        # If split empty on one side → skip
        if len(paths_train_c) == 0 or len(paths_test_c) == 0:
            new_train_paths.extend(paths_train_c)
            new_train_labels.extend(labels_train_c)
            new_test_paths.extend(paths_test_c)
            new_test_labels.extend(labels_test_c)
            continue

        # ---- Combine into unified class list ----
        all_paths_c  = np.concatenate([paths_train_c, paths_test_c])
        all_labels_c = np.concatenate([labels_train_c, labels_test_c])

        n_train_c = len(paths_train_c)
        n_test_c  = len(paths_test_c)

        # Indices in this unified array
        idx_train_c = np.arange(0, n_train_c)
        idx_test_c  = np.arange(n_train_c, n_train_c + n_test_c)

        # ---- Compute swap size ----
        s_c = int(np.floor(zeta * n_train_c))

        if s_c == 0:
            # no refinement
            new_train_paths.extend(paths_train_c)
            new_train_labels.extend(labels_train_c)
            new_test_paths.extend(paths_test_c)
            new_test_labels.extend(labels_test_c)
            continue

        # ---- Select swap indices ----
        swap_train = np.random.choice(idx_train_c, size=s_c, replace=False)
        swap_test  = np.random.choice(idx_test_c, size=s_c, replace=False)

        # ---- Remove swap samples ----
        keep_train = np.setdiff1d(idx_train_c, swap_train)
        keep_test  = np.setdiff1d(idx_test_c, swap_test)

        # ---- Perform the swap ----
        new_idx_train = np.concatenate([keep_train, swap_test])
        new_idx_test  = np.concatenate([keep_test,  swap_train])

        # ---- Reconstruct splits for this class ----
        final_train_paths  = all_paths_c[new_idx_train]
        final_train_labels = all_labels_c[new_idx_train]
        final_test_paths   = all_paths_c[new_idx_test]
        final_test_labels  = all_labels_c[new_idx_test]

        # ---- Append to global lists ----
        new_train_paths.extend(final_train_paths)
        new_train_labels.extend(final_train_labels)
        new_test_paths.extend(final_test_paths)
        new_test_labels.extend(final_test_labels)

    # Replace originals after all classes processed
    train_paths  = list(new_train_paths)
    train_labels = list(new_train_labels)
    test_paths   = list(new_test_paths)
    test_labels  = list(new_test_labels)

if SHIFT_MAGNITUDE > 0:
    # -----------------------------------------
    # Recompute per-class JSD after refinement
    # -----------------------------------------

    total_jsd_score = 0
    unique_classes = np.unique(labels)

    for cls in unique_classes:

        # all global indices belonging to class cls
        cls_mask = labels == cls
        cls_emb = embeddings[cls_mask]
        cls_paths = paths[cls_mask]

        # which of these belong to train?
        cls_train_mask = np.isin(cls_paths, train_paths)
        cls_test_mask  = np.isin(cls_paths, test_paths)

        # get embeddings for each split
        emb_train_c = cls_emb[cls_train_mask]
        emb_test_c  = cls_emb[cls_test_mask]

        # sanity: skip if the refinement emptied one side (should not happen)
        if len(emb_train_c) == 0 or len(emb_test_c) == 0:
            continue

        mu_train = emb_train_c.mean(axis=0)
        mu_test  = emb_test_c.mean(axis=0)

        jsd_c = jensenshannon(mu_train, mu_test, base=2)**2
        total_jsd_score += jsd_c

    # normalize by number of classes
    total_jsd_score /= len(unique_classes)

    print(f"→ SHIFT_MAGNITUDE: {SHIFT_MAGNITUDE} | NEW TOTAL JSD: {np.round(total_jsd_score, 5)}")


if VISUALIZER:
    all_train_indices = np.isin(np.arange(len(paths)), [np.where(paths == p)[0][0] for p in train_paths])
    all_test_indices = np.isin(np.arange(len(paths)), [np.where(paths == p)[0][0] for p in test_paths])
    print("\n...Visualizing...")
    visualize_split(
                embeddings,
                labels_encoded,
                split_indices=[all_train_indices, all_test_indices],
                split_names=["train", "test"],
                le=le
            )

print("\n...Storing/Finalizing (saving data into their new shifted dataset)...")
save_split(OUTPUT_DIR, "train", train_paths, train_labels)
save_split(OUTPUT_DIR, "test", test_paths, test_labels)

print(f"→ Train: {len(train_paths)} | Test: {len(test_paths)}")
print(f"\n✅ Distr.Drfted Split saved at: {OUTPUT_DIR}")


VAL_RATIO = 0.3

def split_train_to_val(dataset_path):
    train_dir = Path(dataset_path) / "train"
    val_dir = Path(dataset_path) / "val"

    # Create val root
    val_dir.mkdir(parents=True, exist_ok=True)

    # Iterate over each class folder inside /train
    for class_dir in train_dir.iterdir():
        if not class_dir.is_dir():
            continue  # skip files if any

        images = sorted(class_dir.glob("*"))
        num_total = len(images)

        if num_total == 0:
            continue

        num_val = int(num_total * VAL_RATIO)

        random.seed(42)
        val_images = random.sample(images, num_val)

        # Make corresponding val class directory
        val_class_dir = val_dir / class_dir.name
        val_class_dir.mkdir(parents=True, exist_ok=True)

        # Move selected images to validation folder
        for img_path in val_images:
            shutil.move(str(img_path), val_class_dir / img_path.name)


split_train_to_val(OUTPUT_DIR)
