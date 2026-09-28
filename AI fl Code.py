# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-09-02T17:42:29.429726Z","iopub.execute_input":"2025-09-02T17:42:29.429999Z","iopub.status.idle":"2025-09-02T17:54:03.804309Z","shell.execute_reply.started":"2025-09-02T17:42:29.429977Z","shell.execute_reply":"2025-09-02T17:54:03.803739Z"}}
# 0) Setup: robust copy/convert into Kaggle working dir (handles .nii and .nii.gz)
import os, shutil, sys, subprocess, json, warnings
warnings.filterwarnings('ignore')

import nibabel as nib
from glob import glob
from tqdm import tqdm

# Input directories (Kaggle BraTS 2020)
INPUT_ROOT = "/kaggle/input/brats20-dataset-training-validation"
DIR_TRAIN = os.path.join(INPUT_ROOT, "BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData")
DIR_VAL   = os.path.join(INPUT_ROOT, "BraTS2020_ValidationData/MICCAI_BraTS2020_ValidationData")

# Output base (working directory)
WORK_ROOT = "/kaggle/working/brats20_nii_gz"
os.makedirs(WORK_ROOT, exist_ok=True)

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def rel_to_input(p):
    return os.path.relpath(p, INPUT_ROOT)

def safe_copy(src, dst):
    ensure_dir(os.path.dirname(dst))
    if not os.path.exists(dst):
        shutil.copy2(src, dst)

def convert_or_copy_tree(base_dir, out_base):
    if not os.path.exists(base_dir):
        print(f"Skip: missing directory {base_dir}")
        return
    # 1) Handle .nii -> .nii.gz
    nii_files = sorted(glob(os.path.join(base_dir, "*", "*.nii")))
    if len(nii_files) > 0:
        print(f"Converting .nii -> .nii.gz from: {base_dir} (found {len(nii_files)})")
        for niip in tqdm(nii_files):
            img = nib.load(niip)
            rel = rel_to_input(niip) + ".gz"
            outp = os.path.join(out_base, rel)
            ensure_dir(os.path.dirname(outp))
            nib.save(img, outp)
    else:
        print(f"No .nii in {base_dir} (OK).")

    # 2) Copy over existing .nii.gz to working dir
    gz_files = sorted(glob(os.path.join(base_dir, "*", "*.nii.gz")))
    print(f"Copying .nii.gz from: {base_dir} (found {len(gz_files)})")
    for gzp in tqdm(gz_files):
        rel = rel_to_input(gzp)
        outp = os.path.join(out_base, rel)
        if not os.path.exists(outp):
            ensure_dir(os.path.dirname(outp))
            shutil.copy2(gzp, outp)

convert_or_copy_tree(DIR_TRAIN, WORK_ROOT)
convert_or_copy_tree(DIR_VAL,   WORK_ROOT)
print("✅ Data prepared under:", WORK_ROOT)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-09-02T17:54:14.955188Z","iopub.execute_input":"2025-09-02T17:54:14.955998Z","iopub.status.idle":"2025-09-02T17:54:14.974509Z","shell.execute_reply.started":"2025-09-02T17:54:14.955972Z","shell.execute_reply":"2025-09-02T17:54:14.973911Z"}}
# 1) Rename segmentation files containing "Segm" to *_seg.nii.gz in working tree (train + validation)
import os

def rename_segm(root_subdir):
    base = os.path.join(WORK_ROOT, root_subdir)
    if not os.path.exists(base):
        return
    for case_folder in sorted(os.listdir(base)):
        case_path = os.path.join(base, case_folder)
        if not os.path.isdir(case_path): 
            continue
        for file in os.listdir(case_path):
            if file.endswith(".nii.gz") and "Segm" in file:
                old_path = os.path.join(case_path, file)
                new_name = f"{case_folder}_seg.nii.gz"
                new_path = os.path.join(case_path, new_name)
                if not os.path.exists(new_path):
                    os.rename(old_path, new_path)

rename_segm("BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData")
rename_segm("BraTS2020_ValidationData/MICCAI_BraTS2020_ValidationData")
print("✅ Segmentation files standardized (if any 'Segm' existed).")

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-09-02T18:34:03.737073Z","iopub.execute_input":"2025-09-02T18:34:03.737631Z","iopub.status.idle":"2025-09-02T18:34:03.773448Z","shell.execute_reply.started":"2025-09-02T18:34:03.737607Z","shell.execute_reply":"2025-09-02T18:34:03.772884Z"}}
# 2) Core imports, setup, seeds, device
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

try:
    from torchinfo import summary as torchinfo_summary
except Exception:
    torchinfo_summary = None

from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_palette("husl")
plt.style.use('ggplot')

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
set_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    torch.cuda.empty_cache()
    print(f"Using {torch.cuda.get_device_name(0)} ({num_gpus} GPU(s))")
else:
    print("Using CPU")

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-09-02T18:34:07.039468Z","iopub.execute_input":"2025-09-02T18:34:07.040064Z","iopub.status.idle":"2025-09-02T18:34:07.044031Z","shell.execute_reply.started":"2025-09-02T18:34:07.040040Z","shell.execute_reply":"2025-09-02T18:34:07.043352Z"}}
# 3) Diagram-aligned PRE-PROCESSING CONFIG
PREPROC = {
    "use_center_slices": True,     # True => take middle 40 automatically
    "slice_range": (60, 100),      # used if use_center_slices=False
    "resize_to": None,             # e.g., (160,160,40) else None (original 240x240x40)
    "augment_train": True          # random flips/rotations/intensity jitter
}
print("Preproc config:", PREPROC)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-09-02T18:34:09.324430Z","iopub.execute_input":"2025-09-02T18:34:09.324658Z","iopub.status.idle":"2025-09-02T18:34:09.334555Z","shell.execute_reply.started":"2025-09-02T18:34:09.324640Z","shell.execute_reply":"2025-09-02T18:34:09.333984Z"}}
# 4) Split train/val/test (from working TrainingData)
DATA_DIR = "/kaggle/working/brats20_nii_gz/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"
all_patients = sorted([p for p in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, p))])

random.seed(42)
random.shuffle(all_patients)

n_total = len(all_patients)
n_train = int(0.7 * n_total)
n_val   = int(0.15 * n_total)
n_test  = n_total - n_train - n_val

train_patients = all_patients[:n_train]
val_patients   = all_patients[n_train:n_train+n_val]
test_patients  = all_patients[n_train+n_val:]

with open("/kaggle/working/train.txt", "w") as f: f.write("\n".join(train_patients))
with open("/kaggle/working/val.txt", "w") as f:   f.write("\n".join(val_patients))
with open("/kaggle/working/test.txt", "w") as f:  f.write("\n".join(test_patients))

print(f"✅ Split: {len(train_patients)} train, {len(val_patients)} val, {len(test_patients)} test.")

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-09-02T18:34:11.350390Z","iopub.execute_input":"2025-09-02T18:34:11.350654Z","iopub.status.idle":"2025-09-02T18:34:11.356054Z","shell.execute_reply.started":"2025-09-02T18:34:11.350633Z","shell.execute_reply":"2025-09-02T18:34:11.355380Z"}}
# 5) Colors + helpers used across notebook
SEG_COLORS = {
    0: (0, 0, 0),
    1: (255, 0, 0),     # Necrosis
    2: (0, 255, 0),     # Edema
    3: (255, 255, 0)    # Active/Enhancing tumor
}

def label_to_color(seg_slice):
    h, w = seg_slice.shape
    color_img = np.zeros((h, w, 3), dtype=np.uint8)
    for label, color in SEG_COLORS.items():
        color_img[seg_slice == label] = color
    return color_img

def read_patients(filename):
    with open(filename, "r") as f:
        return f.read().splitlines()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-09-02T18:34:14.495978Z","iopub.execute_input":"2025-09-02T18:34:14.496692Z","iopub.status.idle":"2025-09-02T18:34:14.509714Z","shell.execute_reply.started":"2025-09-02T18:34:14.496665Z","shell.execute_reply":"2025-09-02T18:34:14.508989Z"}}
# 6) Dataset with normalization + center slices + optional resize + augmentation
import torch.nn.functional as Fnn
import nibabel as nib

class BraTSDataset(Dataset):
    def __init__(self, patient_list, data_dir, slice_range=(60, 100),
                 use_center_slices=False, center_count=40,
                 augment=False, resize_to=None):
        self.data_dir = data_dir
        self.patients = patient_list
        self.modalities = ["flair", "t1", "t1ce", "t2"]
        self.slice_range = slice_range
        self.use_center_slices = use_center_slices
        self.center_count = center_count
        self.augment = augment
        self.resize_to = resize_to

    def load_nifti(self, path):
        return np.array(nib.load(path).get_fdata(), dtype=np.float32)

    def _compute_center_range(self, depth):
        cnt = self.center_count
        z0 = max(0, (depth - cnt) // 2)
        z1 = z0 + cnt
        return (z0, z1)

    def __getitem__(self, idx):
        pid = self.patients[idx]
        p_path = os.path.join(self.data_dir, pid)

        # Determine D for center slicing
        flair_full = self.load_nifti(os.path.join(p_path, f"{pid}_flair.nii.gz"))
        H, W, D = flair_full.shape
        if self.use_center_slices:
            z0, z1 = self._compute_center_range(D)
        else:
            z0, z1 = self.slice_range

        imgs = []
        for mod in self.modalities:
            vol = self.load_nifti(os.path.join(p_path, f"{pid}_{mod}.nii.gz"))
            vol = vol[:, :, z0:z1]
            mu, sd = np.mean(vol), np.std(vol) + 1e-6
            vol = (vol - mu) / sd
            imgs.append(np.nan_to_num(vol, copy=False))

        imgs = torch.tensor(np.stack(imgs), dtype=torch.float32)  # [C,H,W,D]

        seg = self.load_nifti(os.path.join(p_path, f"{pid}_seg.nii.gz")).astype(np.uint8)
        seg = seg[:, :, z0:z1]
        seg[seg == 4] = 3
        seg = torch.tensor(seg, dtype=torch.long)                 # [H,W,D]

        # Optional resizing to (H,W,D) = (Ht,Wt,Dt)
        if self.resize_to is not None:
            Ht, Wt, Dt = self.resize_to
            imgs5 = imgs.permute(0,3,1,2).unsqueeze(0)            # [1,C,D,H,W]
            imgs5 = Fnn.interpolate(imgs5, size=(Dt, Ht, Wt), mode="trilinear", align_corners=False)
            imgs  = imgs5.squeeze(0).permute(0,2,3,1)             # [C,Ht,Wt,Dt]

            seg5  = seg.permute(2,0,1).unsqueeze(0).unsqueeze(0).float()  # [1,1,D,H,W]
            seg5  = Fnn.interpolate(seg5, size=(Dt, Ht, Wt), mode="nearest")
            seg   = seg5.squeeze(0).squeeze(0).permute(1,2,0).long()      # [Ht,Wt,Dt]

        # Augmentation (train only)
        if self.augment:
            # Horizontal (W)
            if random.random() < 0.5:
                imgs = torch.flip(imgs, dims=[2]); seg = torch.flip(seg, dims=[1])
            # Vertical (H)
            if random.random() < 0.5:
                imgs = torch.flip(imgs, dims=[1]); seg = torch.flip(seg, dims=[0])
            # Rot90 in-plane
            if random.random() < 0.5:
                k = random.choice([1,2,3])
                imgs = torch.rot90(imgs, k=k, dims=(1,2))
                seg  = torch.rot90(seg,  k=k, dims=(0,1))
            # Intensity jitter
            if random.random() < 0.5:
                scale = 1.0 + 0.1*(2*random.random()-1)
                shift = 0.05*(2*random.random()-1)
                imgs = imgs*scale + shift

        return imgs, seg

    def __len__(self):
        return len(self.patients)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-09-02T18:34:18.159575Z","iopub.execute_input":"2025-09-02T18:34:18.159842Z","iopub.status.idle":"2025-09-02T18:34:18.886856Z","shell.execute_reply.started":"2025-09-02T18:34:18.159822Z","shell.execute_reply":"2025-09-02T18:34:18.886138Z"}}
# 7) Create datasets/loaders using PREPROC
train_dataset = BraTSDataset(
    read_patients("/kaggle/working/train.txt"), DATA_DIR,
    slice_range=PREPROC["slice_range"],
    use_center_slices=PREPROC["use_center_slices"],
    augment=PREPROC["augment_train"],
    resize_to=PREPROC["resize_to"]
)
val_dataset = BraTSDataset(
    read_patients("/kaggle/working/val.txt"), DATA_DIR,
    slice_range=PREPROC["slice_range"],
    use_center_slices=PREPROC["use_center_slices"],
    augment=False,
    resize_to=PREPROC["resize_to"]
)
test_dataset = BraTSDataset(
    read_patients("/kaggle/working/test.txt"), DATA_DIR,
    slice_range=PREPROC["slice_range"],
    use_center_slices=PREPROC["use_center_slices"],
    augment=False,
    resize_to=PREPROC["resize_to"]
)

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True,  num_workers=0, pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
test_loader  = DataLoader(test_dataset,  batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

sample_img, sample_mask = next(iter(train_loader))
print("Sample shapes:", sample_img.shape, sample_mask.shape)
del sample_img, sample_mask

# %% [code] {"execution":{"iopub.status.busy":"2025-09-02T18:36:03.097414Z","iopub.execute_input":"2025-09-02T18:36:03.097692Z","iopub.status.idle":"2025-09-02T18:36:03.363621Z","shell.execute_reply.started":"2025-09-02T18:36:03.097672Z","shell.execute_reply":"2025-09-02T18:36:03.362971Z"}}
# %% [code]
# Preprocessing Preview: ONLY Original vs Normalized (auto-runs + supports specific patient/slice)

import os, random
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

# Safe fallbacks if globals not defined yet
DATA_DIR = globals().get(
    "DATA_DIR",
    "/kaggle/working/brats20_nii_gz/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"
)
PREPROC = globals().get("PREPROC", {"use_center_slices": True, "slice_range": (60, 100)})

def _safe_read_patients(txt_path):
    try:
        if os.path.exists(txt_path):
            with open(txt_path, "r") as f:
                return [l.strip() for l in f if l.strip()]
    except:
        pass
    # Fallback: list from DATA_DIR
    return sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])

def preview_preprocessing(modality="flair", set_name="train",
                          specific_pid=None, specific_z=None):
    """
    Display:
      - Original slice (no normalization)
      - Normalized slice (z-score over 40-slice window)

    Selection:
      - specific_pid (optional) -> use that case
      - specific_z (optional)   -> window centered on this slice (40 slices)
      - else center-40 (if PREPROC['use_center_slices']=True), otherwise PREPROC['slice_range']
    """
    # 1) Pick set and patients list
    set_name_l = set_name.lower()
    if set_name_l == "train":
        patients = _safe_read_patients("/kaggle/working/train.txt")
    elif set_name_l == "val":
        patients = _safe_read_patients("/kaggle/working/val.txt")
    else:
        patients = _safe_read_patients("/kaggle/working/test.txt")

    if len(patients) == 0:
        print(f"No patients in {set_name} set. Check splits or DATA_DIR.")
        return

    # 2) Choose patient
    if specific_pid is not None and os.path.isdir(os.path.join(DATA_DIR, specific_pid)):
        pid = specific_pid
    else:
        if specific_pid is not None:
            print(f"Specific patient not found in DATA_DIR: {specific_pid} (using random instead)")
        pid = random.choice(patients)

    case_path = os.path.join(DATA_DIR, pid)
    vol_path  = os.path.join(case_path, f"{pid}_{modality}.nii.gz")
    if not os.path.exists(vol_path):
        print(f"Missing file: {vol_path}")
        return

    # 3) Load full original volume
    img = nib.load(vol_path)
    vol = img.get_fdata().astype(np.float32)   # [H, W, D]
    H, W, D = vol.shape

    # 4) Decide slice and normalization window (40 slices)
    if specific_z is not None:
        z_global = int(np.clip(specific_z, 0, D-1))
        half = 20
        z0 = max(0, z_global - half)
        z1 = min(D, z0 + 40)
        z0 = max(0, z1 - 40)   # adjust if near end
        z_local = z_global - z0
    else:
        if PREPROC.get("use_center_slices", True):
            z0 = max(0, (D - 40) // 2)
            z1 = min(D, z0 + 40)
            z_global = (z0 + z1) // 2
            z_local = z_global - z0
        else:
            z0, z1 = PREPROC.get("slice_range", (60, 100))
            z0 = max(0, min(z0, D - 1))
            z1 = max(z0 + 1, min(z1, D))
            z_global = (z0 + z1) // 2
            z_local = z_global - z0

    # 5) Slices
    orig_slice = vol[:, :, z_global]
    sub = vol[:, :, z0:z1]
    mu = float(np.mean(sub)); sd = float(np.std(sub) + 1e-6)
    norm_slice = ((sub - mu) / sd)[:, :, z_local]

    print(f"Preview -> set={set_name}, pid={pid}, modality={modality}, D={D}, "
          f"norm_window=({z0},{z1}), z_global={z_global}, mean={mu:.2f}, sd={sd:.2f}")

    # 6) Plot Original vs Normalized
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(orig_slice, cmap="gray")
    axes[0].set_title(f"{pid} | {modality.upper()} Original (z={z_global})")
    axes[0].axis("off")

    axes[1].imshow(norm_slice, cmap="gray")
    axes[1].set_title("Normalized (z-score)")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()

    return pid, z_global

# ====== CONFIG for auto-preview (you can edit these) ======
PREVIEW_SET       = "train"  # "train" | "val" | "test"
PREVIEW_MODALITY  = "flair"  # "flair" | "t1" | "t1ce" | "t2"
PREVIEW_PID       = None     # e.g., "BraTS20_Training_177" or None for random
PREVIEW_Z         = None     # e.g., 77 or None for middle of window

# Auto-run preview
preview_preprocessing(
    modality=PREVIEW_MODALITY,
    set_name=PREVIEW_SET,
    specific_pid=PREVIEW_PID,
    specific_z=PREVIEW_Z
)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-09-02T17:55:25.552458Z","iopub.execute_input":"2025-09-02T17:55:25.552724Z","iopub.status.idle":"2025-09-02T17:55:28.596222Z","shell.execute_reply.started":"2025-09-02T17:55:25.552705Z","shell.execute_reply":"2025-09-02T17:55:28.595485Z"}}
# 8) Quick visualization: random examples + Original vs GT comparison
def show_random_visuals(patients, data_dir, title="Train", num_cases=3, use_center=True, slice_range=(60,100)):
    modalities = ["flair", "t1", "t1ce", "t2"]
    if len(patients) == 0:
        print(f"No patients in {title}"); return
    selected = random.sample(patients, min(num_cases, len(patients)))

    plt.figure(figsize=(18, 6*len(selected)))
    for idx, case_id in enumerate(selected):
        case_path = os.path.join(data_dir, case_id)
        images, seg = [], None
        for mod in modalities:
            img = nib.load(os.path.join(case_path, f"{case_id}_{mod}.nii.gz")).get_fdata()
            images.append(img)
        seg = nib.load(os.path.join(case_path, f"{case_id}_seg.nii.gz")).get_fdata().astype(np.uint8)
        seg[seg==4] = 3

        D = images[0].shape[2]
        if use_center:
            z0 = max(0, (D-40)//2); z1 = z0+40; sl = (z0+z1)//2
        else:
            sl = random.randint(*slice_range)

        for i, mod_img in enumerate(images):
            plt.subplot(len(selected), 5, idx*5 + i + 1)
            plt.imshow(mod_img[:,:,sl], cmap='gray')
            plt.title(f"{case_id} - {modalities[i].upper()}")
            plt.axis('off')
        plt.subplot(len(selected), 5, idx*5 + 5)
        plt.imshow(label_to_color(seg[:,:,sl]))
        plt.title("SEG (color)")
        plt.axis('off')
    plt.suptitle(f"Random examples | {title}", fontsize=16)
    plt.tight_layout(); plt.show()

def show_original_vs_gt(patients, data_dir, use_center=True, slice_range=(60,100)):
    if len(patients)==0: 
        print("No patients"); return
    pid = random.choice(patients)
    p_path = os.path.join(data_dir, pid)
    flair = nib.load(os.path.join(p_path, f"{pid}_flair.nii.gz")).get_fdata()
    seg   = nib.load(os.path.join(p_path, f"{pid}_seg.nii.gz")).get_fdata().astype(np.uint8)
    seg[seg==4] = 3
    D = flair.shape[2]
    if use_center:
        z0 = max(0, (D-40)//2); z1 = z0+40; z = (z0+z1)//2
    else:
        z = random.randint(*slice_range)

    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1); plt.imshow(flair[:,:,z], cmap='gray'); plt.title(f"{pid} | FLAIR z={z}"); plt.axis('off')
    plt.subplot(1,2,2); plt.imshow(flair[:,:,z], cmap='gray'); plt.imshow(label_to_color(seg[:,:,z]), alpha=0.45)
    plt.title("Ground Truth Overlay"); plt.axis('off'); plt.tight_layout(); plt.show()

train_p = read_patients("/kaggle/working/train.txt")
val_p   = read_patients("/kaggle/working/val.txt")
test_p  = read_patients("/kaggle/working/test.txt")
show_random_visuals(train_p, DATA_DIR, title="Train")
show_original_vs_gt(train_p, DATA_DIR, use_center=True)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-09-02T17:55:53.282536Z","iopub.execute_input":"2025-09-02T17:55:53.282815Z","iopub.status.idle":"2025-09-02T17:55:53.299014Z","shell.execute_reply.started":"2025-09-02T17:55:53.282794Z","shell.execute_reply":"2025-09-02T17:55:53.298181Z"}}
# 9) Model (3D U-Net with residual blocks) + Loss + Metrics
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, padding=1)
        self.bn1   = nn.BatchNorm3d(out_channels)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1)
        self.bn2   = nn.BatchNorm3d(out_channels)
        self.residual = nn.Conv3d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        identity = self.residual(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)

class Improved3DUNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=4, base_filters=16):
        super().__init__()
        self.enc1 = ResidualBlock(in_channels, base_filters)
        self.pool1 = nn.MaxPool3d(2)
        self.enc2 = ResidualBlock(base_filters, base_filters*2)
        self.pool2 = nn.MaxPool3d(2)

        self.bottleneck = ResidualBlock(base_filters*2, base_filters*4)

        self.up2  = nn.ConvTranspose3d(base_filters*4, base_filters*2, 2, stride=2)
        self.dec2 = ResidualBlock(base_filters*4, base_filters*2)
        self.up1  = nn.ConvTranspose3d(base_filters*2, base_filters, 2, stride=2)
        self.dec1 = ResidualBlock(base_filters*2, base_filters)
        self.out_conv = nn.Conv3d(base_filters, out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b  = self.bottleneck(self.pool2(e2))
        d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out_conv(d1)

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth
    def forward(self, y_pred, y_true):
        y_pred = F.softmax(y_pred, dim=1)
        y_true_oh = F.one_hot(y_true, num_classes=y_pred.shape[1]).permute(0,4,1,2,3).float()
        y_pred = y_pred.view(y_pred.size(0), y_pred.size(1), -1)
        y_true_oh = y_true_oh.view(y_true_oh.size(0), y_true_oh.size(1), -1)
        intersection = (y_pred*y_true_oh).sum(dim=2)
        dice = (2.*intersection + self.smooth)/(y_pred.sum(dim=2)+y_true_oh.sum(dim=2)+self.smooth)
        return 1 - dice.mean()

class CombinedLoss(nn.Module):
    def __init__(self, weight_dice=0.7, weight_ce=0.3):
        super().__init__()
        self.dice = DiceLoss(); self.ce = nn.CrossEntropyLoss()
        self.wd = weight_dice; self.wc = weight_ce
    def forward(self, inp, tgt):
        return self.wc*self.ce(inp, tgt) + self.wd*self.dice(inp, tgt)

criterion = CombinedLoss(0.7, 0.3)

def dice_coefficient(y_pred, y_true, smooth=1e-6):
    num_classes = y_pred.shape[1]
    y_pred_soft = F.softmax(y_pred.float(), dim=1)
    y_true_oh = F.one_hot(y_true, num_classes=num_classes).permute(0,4,1,2,3).float()
    y_pred_flat = y_pred_soft.view(y_pred_soft.size(0), num_classes, -1)
    y_true_flat = y_true_oh.view(y_true_oh.size(0), num_classes, -1)
    inter = (y_pred_flat*y_true_flat).sum(dim=2)
    union = y_pred_flat.sum(dim=2) + y_true_flat.sum(dim=2)
    dice = (2.*inter + smooth)/(union + smooth)
    return dice.mean().item()

def iou_and_accuracy(y_pred, y_true, num_classes=4, ignore_background=True, smooth=1e-6):
    y_lab = torch.argmax(y_pred, dim=1)
    class_range = range(1,num_classes) if ignore_background else range(num_classes)
    ious, accs = [], []
    for cls in class_range:
        p = (y_lab==cls); t = (y_true==cls)
        inter = (p&t).sum().float()
        union = (p|t).sum().float()
        if union>0:
            iou = (inter+smooth)/(union+smooth)
            acc = (p==t).float().mean()
            ious.append(iou); accs.append(acc)
    miou = torch.mean(torch.stack(ious)) if ious else torch.tensor(0.0)
    macc = torch.mean(torch.stack(accs)) if accs else torch.tensor(0.0)
    return miou.item(), macc.item()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-09-02T17:56:03.563071Z","iopub.execute_input":"2025-09-02T17:56:03.563364Z","iopub.status.idle":"2025-09-02T17:56:03.574575Z","shell.execute_reply.started":"2025-09-02T17:56:03.563332Z","shell.execute_reply":"2025-09-02T17:56:03.573968Z"}}
# 10) Train/Val loops
from tqdm.auto import tqdm
def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None, use_amp=True, desc="Train", max_batches=None):
    model.train()
    tot_loss=tot_dice=tot_iou=tot_acc=0.0
    n_total = len(loader) if max_batches is None else min(len(loader), max_batches)
    loop = tqdm(enumerate(loader), total=n_total, desc=desc, leave=False, dynamic_ncols=True)
    for i,(x,y) in loop:
        if max_batches is not None and i>=max_batches: break
        x,y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            out = model(x)
            loss = criterion(out, y)
        if scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
        else:
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        dsc = dice_coefficient(out,y); iou,acc = iou_and_accuracy(out,y)
        tot_loss+=loss.item(); tot_dice+=dsc; tot_iou+=iou; tot_acc+=acc
        loop.set_postfix(loss=f"{loss.item():.4f}", dice=f"{dsc:.4f}", iou=f"{iou:.4f}", acc=f"{acc:.4f}")
    n = n_total if n_total>0 else 1
    return tot_loss/n, tot_dice/n, tot_iou/n, tot_acc/n

@torch.no_grad()
def validate_one_epoch(model, loader, criterion, device, use_amp=True, desc="Val", max_batches=None):
    model.eval()
    tot_loss=tot_dice=tot_iou=tot_acc=0.0
    n_total = len(loader) if max_batches is None else min(len(loader), max_batches)
    loop = tqdm(enumerate(loader), total=n_total, desc=desc, leave=False, dynamic_ncols=True)
    for i,(x,y) in loop:
        if max_batches is not None and i>=max_batches: break
        x,y = x.to(device), y.to(device)
        with torch.cuda.amp.autocast(enabled=use_amp):
            out = model(x); loss = criterion(out,y)
        dsc = dice_coefficient(out,y); iou,acc = iou_and_accuracy(out,y)
        tot_loss+=loss.item(); tot_dice+=dsc; tot_iou+=iou; tot_acc+=acc
        loop.set_postfix(loss=f"{loss.item():.4f}", dice=f"{dsc:.4f}", iou=f"{iou:.4f}", acc=f"{acc:.4f}")
    n = n_total if n_total>0 else 1
    return tot_loss/n, tot_dice/n, tot_iou/n, tot_acc/n

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001):
        self.patience=patience; self.min_delta=min_delta
        self.counter=0; self.best=float('inf'); self.stop=False
    def step(self, val_loss):
        if val_loss < self.best - self.min_delta:
            self.best=val_loss; self.counter=0
        else:
            self.counter+=1
            if self.counter>=self.patience: self.stop=True

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-09-02T17:56:20.798952Z","iopub.execute_input":"2025-09-02T17:56:20.799183Z","iopub.status.idle":"2025-09-02T17:56:20.804988Z","shell.execute_reply.started":"2025-09-02T17:56:20.799166Z","shell.execute_reply":"2025-09-02T17:56:20.804409Z"}}
# 11) FL config
FL_CONFIG = {
    "num_rounds": 1,
    "local_epochs": 1,
    "num_clients": 1,       # set >1 to simulate multi-client
    "use_amp": True,
    "lr": 2e-4,
    "weight_decay": 1e-5,
    "scheduler_Tmax": 30,
    "scheduler_eta_min": 1e-6,
    "save_dir": "/kaggle/working/fl",
    "debug_max_batches": None
}
os.makedirs(FL_CONFIG["save_dir"], exist_ok=True)
with open(os.path.join(FL_CONFIG["save_dir"], "fl_config.json"), "w") as f: json.dump(FL_CONFIG, f, indent=2)
print("FL config:", FL_CONFIG)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-09-02T17:56:26.754304Z","iopub.execute_input":"2025-09-02T17:56:26.754575Z","iopub.status.idle":"2025-09-02T17:56:26.858492Z","shell.execute_reply.started":"2025-09-02T17:56:26.754556Z","shell.execute_reply":"2025-09-02T17:56:26.857905Z"}}
# 12) Blockchain (with encryption support)
import hashlib
from datetime import datetime

def sha256_bytes(b:bytes)->str: return hashlib.sha256(b).hexdigest()
def file_sha256(path:str)->str:
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for ch in iter(lambda:f.read(8192), b""): h.update(ch)
    return h.hexdigest()

def state_dict_hash(state:dict, as_dtype=torch.float32)->str:
    h=hashlib.sha256()
    for k in sorted(state.keys()):
        v=state[k]
        h.update(k.encode('utf-8'))
        h.update(v.detach().cpu().to(as_dtype).contiguous().numpy().tobytes())
    return h.hexdigest()

class SimpleBlockchain:
    def __init__(self, save_dir="/kaggle/working/blockchain", difficulty=0):
        self.chain=[]; self.save_dir=save_dir; self.difficulty=difficulty
        os.makedirs(save_dir, exist_ok=True)
    def _block(self, index, prev_hash, payload):
        return {"index":index,"timestamp":datetime.utcnow().isoformat()+"Z","prev_hash":prev_hash,
                "payload":payload,"nonce":0,"hash":""}
    def _compute_hash(self, blk): 
        b=json.dumps({k:blk[k] for k in blk if k!="hash"}, sort_keys=True).encode("utf-8")
        return sha256_bytes(b)
    def _mine(self, blk):
        if self.difficulty<=0: blk["hash"]=self._compute_hash(blk); return blk
        pref="0"*self.difficulty
        while True:
            blk["hash"]=self._compute_hash(blk)
            if blk["hash"].startswith(pref): return blk
            blk["nonce"]+=1
    def create_genesis(self):
        if self.chain: return
        g=self._block(0,"0"*64,{"type":"genesis","msg":"Start FL+Blockchain"})
        g=self._mine(g); self.chain.append(g); self._persist_block(g); self._persist_ledger()
    def add_block(self,payload):
        prev=self.chain[-1]["hash"] if self.chain else "0"*64
        b=self._block(len(self.chain), prev, payload)
        b=self._mine(b); self.chain.append(b); self._persist_block(b); self._persist_ledger(); return b
    def is_valid(self):
        for i,blk in enumerate(self.chain):
            comp=self._compute_hash(blk)
            if comp!=blk["hash"]: print("Invalid hash at",i); return False
            if i>0 and blk["prev_hash"]!=self.chain[i-1]["hash"]: print("Broken link at",i); return False
            if self.difficulty>0 and not blk["hash"].startswith("0"*self.difficulty): print("PoW violated at",i); return False
        return True
    def _persist_block(self,blk):
        with open(os.path.join(self.save_dir, f"block_{blk['index']}.json"),"w") as f: json.dump(blk,f,indent=2)
    def _persist_ledger(self):
        led={"difficulty":self.difficulty,"length":len(self.chain),"chain":self.chain}
        with open(os.path.join(self.save_dir,"ledger.json"),"w") as f: json.dump(led,f,indent=2)

BC_SAVE_DIR="/kaggle/working/blockchain"
blockchain=SimpleBlockchain(save_dir=BC_SAVE_DIR, difficulty=0)
blockchain.create_genesis()
print("Blockchain initialized:", BC_SAVE_DIR)

# Encryption helper
def _import_fernet():
    try:
        from cryptography.fernet import Fernet
        return Fernet
    except Exception:
        subprocess.run([sys.executable,"-m","pip","install","-q","cryptography"], check=False)
        from cryptography.fernet import Fernet
        return Fernet

Fernet = _import_fernet()

def encrypt_ckpt(ckpt_path, save_dir="/kaggle/working/fl", round_idx=None):
    key = Fernet.generate_key()
    f = Fernet(key)
    with open(ckpt_path,"rb") as fin: data=fin.read()
    enc = f.encrypt(data)
    enc_path = ckpt_path + ".enc"
    with open(enc_path,"wb") as fout: fout.write(enc)
    if round_idx is not None:
        with open(os.path.join(save_dir, f"round_{round_idx}_key.txt"), "wb") as fk:
            fk.write(key)  # demo only; don't store keys alongside artifacts in real settings
    return enc_path, key

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-09-02T17:56:32.155784Z","iopub.execute_input":"2025-09-02T17:56:32.156484Z","iopub.status.idle":"2025-09-02T17:56:32.168186Z","shell.execute_reply.started":"2025-09-02T17:56:32.156458Z","shell.execute_reply":"2025-09-02T17:56:32.167322Z"}}
# 13) FL utilities (build model, FedAvg, multi-client builder)
from copy import deepcopy

def build_model():
    return Improved3DUNet(in_channels=4, out_channels=4, base_filters=16).to(device)

def get_state_dict(model):
    return {k:v.detach().cpu() for k,v in model.state_dict().items()}

def set_state_dict(model, state):
    model.load_state_dict(state)

def fedavg(state_list, weight_list):
    total=float(sum(weight_list))
    agg=deepcopy(state_list[0])
    for k in agg.keys():
        agg[k]=agg[k]*(weight_list[0]/total)
    for i in range(1,len(state_list)):
        for k in agg.keys():
            agg[k]+=state_list[i][k]*(weight_list[i]/total)
    return agg

class SimpleClient:
    def __init__(self, train_loader, val_loader, client_id=0):
        self.train_loader=train_loader; self.val_loader=val_loader; self.client_id=client_id
    def train(self, global_state, config, round_index=0):
        model=build_model(); set_state_dict(model, global_state)
        optimizer=optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
        scaler=GradScaler(enabled=config["use_amp"])
        scheduler=optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["scheduler_Tmax"], eta_min=config["scheduler_eta_min"])
        logs={}
        for e in range(config["local_epochs"]):
            tl,td,tiou,tacc = train_one_epoch(model, self.train_loader, optimizer, criterion, device, scaler, config["use_amp"],
                                              desc=f"R{round_index+1} Train C{self.client_id}", max_batches=config.get("debug_max_batches"))
            vl,vd,viou,vacc = validate_one_epoch(model, self.val_loader, criterion, device, config["use_amp"],
                                                 desc=f"R{round_index+1} Val   C{self.client_id}", max_batches=config.get("debug_max_batches"))
            logs={"train_loss":tl,"train_dice":td,"train_iou":tiou,"train_acc":tacc,
                  "val_loss":vl,"val_dice":vd,"val_iou":viou,"val_acc":vacc}
            scheduler.step()
        updated_state=get_state_dict(model)
        num_samples=len(self.train_loader.dataset)
        lr_used=optimizer.param_groups[0]['lr']
        return updated_state, num_samples, logs, lr_used

import numpy as np
def build_clients_from_splits(num_clients:int):
    if num_clients<=1:
        return [SimpleClient(train_loader, val_loader, client_id=0)]
    tr = read_patients("/kaggle/working/train.txt")
    va = read_patients("/kaggle/working/val.txt")
    shards_tr = np.array_split(tr, num_clients)
    shards_va = np.array_split(va, num_clients)
    client_list=[]
    for i in range(num_clients):
        ds_tr = BraTSDataset(list(shards_tr[i]), DATA_DIR,
                             slice_range=PREPROC["slice_range"],
                             use_center_slices=PREPROC["use_center_slices"],
                             augment=PREPROC["augment_train"],
                             resize_to=PREPROC["resize_to"])
        ds_va = BraTSDataset(list(shards_va[i]), DATA_DIR,
                             slice_range=PREPROC["slice_range"],
                             use_center_slices=PREPROC["use_center_slices"],
                             augment=False, resize_to=PREPROC["resize_to"])
        dl_tr = DataLoader(ds_tr, batch_size=1, shuffle=True,  num_workers=0, pin_memory=True)
        dl_va = DataLoader(ds_va, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
        client_list.append(SimpleClient(dl_tr, dl_va, client_id=i))
    return client_list

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-09-02T17:56:36.926598Z","iopub.execute_input":"2025-09-02T17:56:36.926858Z","iopub.status.idle":"2025-09-02T17:56:38.291604Z","shell.execute_reply.started":"2025-09-02T17:56:36.926837Z","shell.execute_reply":"2025-09-02T17:56:38.291047Z"}}
# 14) Optional: model summary
global_model = build_model()
if torchinfo_summary is not None:
    try:
        tmp_x, _ = next(iter(train_loader))
        _, C, H, W, D = tmp_x.shape
        print("Model summary with input size:", (1, C, H, W, D))
        torchinfo_summary(global_model, input_size=(1, C, H, W, D))
        del tmp_x
    except Exception as e:
        print("Summary skipped:", e)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-09-02T17:56:42.955042Z","iopub.execute_input":"2025-09-02T17:56:42.955296Z","iopub.status.idle":"2025-09-02T18:01:11.559530Z","shell.execute_reply.started":"2025-09-02T17:56:42.955279Z","shell.execute_reply":"2025-09-02T18:01:11.558976Z"}}
# 15) Federated Training + Blockchain + Encrypted checkpoints
clients = build_clients_from_splits(FL_CONFIG["num_clients"])

history = {"round": [], "train_loss": [], "train_dice": [], "train_iou": [], "train_acc": [],
           "val_loss": [], "val_dice": [], "val_iou": [], "val_acc": [], "lr": []}

best_val_dice=-1.0
SAVE_DIR = FL_CONFIG["save_dir"]
best_path  = os.path.join(SAVE_DIR, "global_model_best.pth")
final_path = os.path.join(SAVE_DIR, "global_model_final.pth")

with tqdm(range(FL_CONFIG["num_rounds"]), desc="FL Rounds", unit="round", dynamic_ncols=True) as rounds_bar:
    for r in rounds_bar:
        global_state = get_state_dict(global_model)

        client_states=[]; client_sizes=[]; round_logs=[]; round_lrs=[]; client_hashes=[]
        with tqdm(clients, desc=f"Round {r+1} | Clients", unit="client", leave=False, dynamic_ncols=True) as cbar:
            for client in cbar:
                st,n,logs,lr_used = client.train(global_state, FL_CONFIG, round_index=r)
                client_states.append(st); client_sizes.append(n); round_logs.append(logs); round_lrs.append(lr_used)
                client_hashes.append(state_dict_hash(st))
                cbar.set_postfix(train_dice=f"{logs['train_dice']:.4f}", val_dice=f"{logs['val_dice']:.4f}", lr=f"{lr_used:.1e}")

        new_global_state = fedavg(client_states, client_sizes)
        set_state_dict(global_model, new_global_state)

        ckpt_path = os.path.join(SAVE_DIR, f"global_round_{r+1}.pth")
        torch.save(global_model.state_dict(), ckpt_path)
        ckpt_sha  = file_sha256(ckpt_path)
        enc_path, enc_key = encrypt_ckpt(ckpt_path, save_dir=SAVE_DIR, round_idx=r+1)
        enc_sha = file_sha256(enc_path)
        agg_hash = state_dict_hash(new_global_state)
        tqdm.write(f"Saved: {ckpt_path} sha256={ckpt_sha[:12]}... | enc={os.path.basename(enc_path)}")

        # Aggregate metrics from clients
        mean_train_loss = float(np.mean([l["train_loss"] for l in round_logs]))
        mean_train_dice = float(np.mean([l["train_dice"] for l in round_logs]))
        mean_train_iou  = float(np.mean([l["train_iou"]  for l in round_logs]))
        mean_train_acc  = float(np.mean([l["train_acc"]  for l in round_logs]))
        mean_val_loss   = float(np.mean([l["val_loss"]   for l in round_logs]))
        mean_val_dice   = float(np.mean([l["val_dice"]   for l in round_logs]))
        mean_val_iou    = float(np.mean([l["val_iou"]    for l in round_logs]))
        mean_val_acc    = float(np.mean([l["val_acc"]    for l in round_logs]))
        mean_lr         = float(np.mean(round_lrs))

        history["round"].append(r+1)
        history["train_loss"].append(mean_train_loss)
        history["train_dice"].append(mean_train_dice)
        history["train_iou"].append(mean_train_iou)
        history["train_acc"].append(mean_train_acc)
        history["val_loss"].append(mean_val_loss)
        history["val_dice"].append(mean_val_dice)
        history["val_iou"].append(mean_val_iou)
        history["val_acc"].append(mean_val_acc)
        history["lr"].append(mean_lr)

        if mean_val_dice>best_val_dice:
            best_val_dice=mean_val_dice
            torch.save(global_model.state_dict(), best_path)
            tqdm.write(f"New best (val_dice={best_val_dice:.4f}) -> {best_path}")

        payload = {
            "type":"fl_round", "round":r+1,
            "client_ids":[c.client_id for c in clients],
            "client_sizes": client_sizes,
            "metrics": {"train_loss":mean_train_loss,"train_dice":mean_train_dice,"train_iou":mean_train_iou,"train_acc":mean_train_acc,
                        "val_loss":mean_val_loss,"val_dice":mean_val_dice,"val_iou":mean_val_iou,"val_acc":mean_val_acc,"lr":mean_lr},
            "client_update_hashes": client_hashes,
            "aggregated_state_hash": agg_hash,
            "global_model_ckpt": os.path.basename(ckpt_path),
            "global_model_sha256": ckpt_sha,
            "encrypted_ckpt": os.path.basename(enc_path),
            "encrypted_ckpt_sha256": enc_sha,
            "encryption":"fernet"
        }
        blk = blockchain.add_block(payload)
        tqdm.write(f"Blockchain -> block #{blk['index']} | hash: {blk['hash'][:12]}...")

torch.save(global_model.state_dict(), final_path)
final_sha = file_sha256(final_path)
print(f"✅ Final global model -> {final_path} | sha256: {final_sha[:12]}...")

blockchain.add_block({"type":"final_model","round":FL_CONFIG["num_rounds"],
                      "final_model_path":os.path.basename(final_path), "final_model_sha256":final_sha})
print("Blockchain valid:", blockchain.is_valid())

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-09-02T18:01:50.292779Z","iopub.execute_input":"2025-09-02T18:01:50.293684Z","iopub.status.idle":"2025-09-02T18:01:51.507323Z","shell.execute_reply.started":"2025-09-02T18:01:50.293659Z","shell.execute_reply":"2025-09-02T18:01:51.506611Z"}}
# 16) Curves: Loss/Dice/Acc/LR
plt.figure(figsize=(16,10))
plt.subplot(2,2,1); plt.plot(history["round"],history["train_loss"],label="Train"); plt.plot(history["round"],history["val_loss"],label="Val")
plt.title("Loss"); plt.xlabel("Round"); plt.grid(True); plt.legend()
plt.subplot(2,2,2); plt.plot(history["round"],history["train_dice"],label="Train"); plt.plot(history["round"],history["val_dice"],label="Val")
plt.title("Dice"); plt.xlabel("Round"); plt.grid(True); plt.legend()
plt.subplot(2,2,3); plt.plot(history["round"],history["train_acc"],label="Train"); plt.plot(history["round"],history["val_acc"],label="Val")
plt.title("Accuracy"); plt.xlabel("Round"); plt.grid(True); plt.legend()
plt.subplot(2,2,4); plt.plot(history["round"],history["lr"],color='purple',label="LR")
plt.title("Learning Rate"); plt.xlabel("Round"); plt.grid(True); plt.legend()
plt.tight_layout(); plt.savefig(os.path.join(SAVE_DIR,"fl_training_curves.png")); plt.show()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-09-02T18:02:04.840007Z","iopub.execute_input":"2025-09-02T18:02:04.840250Z","iopub.status.idle":"2025-09-02T18:20:25.862028Z","shell.execute_reply.started":"2025-09-02T18:02:04.840234Z","shell.execute_reply":"2025-09-02T18:20:25.861350Z"}}
# 17) Evaluate on test set: Accuracy, Dice, IoU, HD95 + per-class table/plot
best_path  = os.path.join(SAVE_DIR, "global_model_best.pth")
final_path = os.path.join(SAVE_DIR, "global_model_final.pth")
load_path  = best_path if os.path.exists(best_path) else final_path

infer_model = Improved3DUNet(in_channels=4, out_channels=4, base_filters=16).to(device)
infer_model.load_state_dict(torch.load(load_path, map_location=device))
infer_model.eval()

# HD95 deps
def _import_surface_distance():
    try:
        from surface_distance import metrics as sd_metrics
        return sd_metrics
    except Exception:
        subprocess.run([sys.executable,"-m","pip","install","-q","surface-distance"], check=False)
        from surface_distance import metrics as sd_metrics
        return sd_metrics
sd_metrics = _import_surface_distance()

def hd95_per_class(pred, gt, spacing=(1,1,1), cls=1):
    gt_bin=(gt==cls); pr_bin=(pred==cls)
    if gt_bin.sum()==0 and pr_bin.sum()==0: return 0.0
    if gt_bin.sum()==0 or pr_bin.sum()==0: return float('inf')
    sd = sd_metrics.compute_surface_distances(gt_bin, pr_bin, spacing)
    return float(sd_metrics.compute_robust_hausdorff(sd, 95))

all_preds=[]; all_targets=[]
test_losses=[]; test_dices=[]; test_accuracies=[]; test_ious=[]
hd95_acc={1:[],2:[],3:[]}

with torch.no_grad():
    for x,y in test_loader:
        x,y=x.to(device), y.to(device)
        y_pred=infer_model(x)
        loss=criterion(y_pred,y); test_losses.append(loss.item())
        dsc=dice_coefficient(y_pred,y); iou,acc=iou_and_accuracy(y_pred,y)
        test_dices.append(dsc); test_accuracies.append(acc); test_ious.append(iou)
        preds=torch.argmax(y_pred,dim=1)[0].cpu().numpy().astype(np.uint8)
        gt   =y[0].cpu().numpy().astype(np.uint8)
        for c in [1,2,3]:
            try: hd=hd95_per_class(preds,gt,(1,1,1),c)
            except Exception: hd=float('inf')
            hd95_acc[c].append(hd)
        all_preds.extend(preds.flatten()); all_targets.extend(gt.flatten())

if len(all_targets)>0:
    overall_acc=(np.array(all_preds)==np.array(all_targets)).mean()
    print(f"\nOverall voxel accuracy: {overall_acc:.4f}")
    class_names=["Background","Necrosis","Edema","Active tumor"]
    print("\n=== Classification Report ===")
    print(classification_report(all_targets, all_preds, target_names=class_names, digits=4))
    print("=== Confusion Matrix ===")
    print(confusion_matrix(all_targets, all_preds, labels=[0,1,2,3]))

    # Per-class metrics for the aggregated voxel labels
    def dice_iou_per_class(y_true, y_pred, class_id):
        y_true_bin=(np.array(y_true)==class_id).astype(np.uint8)
        y_pred_bin=(np.array(y_pred)==class_id).astype(np.uint8)
        inter=np.sum(y_true_bin*y_pred_bin)
        union=np.sum(y_true_bin)+np.sum(y_pred_bin)
        iou_denom=np.sum((y_true_bin+y_pred_bin)>0)
        dice=(2.*inter+1e-6)/(union+1e-6)
        iou =(inter+1e-6)/(iou_denom+1e-6)
        acc=(y_true_bin==y_pred_bin).mean() if y_true_bin.size>0 else 0.0
        return dice,iou,acc

    metrics_data=[]
    print("\n=== Segmentation metrics by class ===")
    print(f"{'Class':15s} {'Dice':>8s} {'IoU':>8s} {'Acc':>8s} {'HD95':>10s} {'Prec':>8s} {'Rec':>8s} {'F1':>8s}")
    for i,name in enumerate(class_names):
        d,iou,acc = dice_iou_per_class(all_targets, all_preds, i)
        y_true_bin=(np.array(all_targets)==i).astype(np.uint8)
        y_pred_bin=(np.array(all_preds)==i).astype(np.uint8)
        prec=precision_score(y_true_bin,y_pred_bin,zero_division=0)
        rec =recall_score(y_true_bin,y_pred_bin,zero_division=0)
        f1  =f1_score(y_true_bin,y_pred_bin,zero_division=0)
        hd  = np.mean([v for v in hd95_acc.get(i,[]) if np.isfinite(v)]) if i in [1,2,3] else 0.0
        metrics_data.append({"Class":name,"Dice":d,"IoU":iou,"Accuracy":acc,"HD95":hd,"Precision":prec,"Recall":rec,"F1-score":f1})
        print(f"{name:15s} {d:8.4f} {iou:8.4f} {acc:8.4f} {hd:10.3f} {prec:8.4f} {rec:8.4f} {f1:8.4f}")

    df_metrics=pd.DataFrame(metrics_data)
    df_metrics.to_csv("/kaggle/working/segmentation_metrics.csv", index=False)
    print("\nSaved CSV: /kaggle/working/segmentation_metrics.csv")

    # Bar plot
    plt.figure(figsize=(12,6))
    bar_w=0.12; x=np.arange(len(class_names))
    for k,metric in enumerate(["Dice","IoU","Accuracy","HD95","Precision","Recall","F1-score"]):
        vals=[row[metric] for row in metrics_data]
        if metric=="HD95":
            # scale HD for display (inverse relation); plot as negative bar for visibility
            vals=[min(v,50.0)/50.0 for v in vals]  # normalized (lower is better)
            metric="HD95 (norm, lower better)"
        plt.bar(x+k*bar_w, vals, width=bar_w, label=metric)
    plt.xticks(x+bar_w*3, class_names); plt.ylim(0,1.05); plt.grid(True,ls='--',alpha=0.4)
    plt.legend(ncol=3); plt.title("Per-class metrics (Test)")
    plt.tight_layout(); plt.savefig("/kaggle/working/segmentation_metrics_plot.png"); plt.show()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-09-02T18:20:30.604315Z","iopub.execute_input":"2025-09-02T18:20:30.604788Z","iopub.status.idle":"2025-09-02T18:20:31.631949Z","shell.execute_reply.started":"2025-09-02T18:20:30.604764Z","shell.execute_reply":"2025-09-02T18:20:31.631271Z"}}
# 18) Per-batch test curves
epochs_bt=list(range(1,len(test_losses)+1))
plt.figure(figsize=(16,10))
plt.subplot(2,2,1); plt.plot(epochs_bt,test_losses,'r'); plt.title("Test Loss"); plt.grid(True)
plt.subplot(2,2,2); plt.plot(epochs_bt,test_dices,'g'); plt.title("Test Dice"); plt.grid(True)
plt.subplot(2,2,3); plt.plot(epochs_bt,test_accuracies,'b'); plt.title("Test Accuracy"); plt.grid(True)
plt.subplot(2,2,4); plt.plot(epochs_bt,test_ious,'m'); plt.title("Test IoU"); plt.grid(True)
plt.tight_layout(); plt.savefig("/kaggle/working/test_metrics_plot.png"); plt.show()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-09-02T18:20:40.960357Z","iopub.execute_input":"2025-09-02T18:20:40.960909Z","iopub.status.idle":"2025-09-02T18:20:41.181309Z","shell.execute_reply.started":"2025-09-02T18:20:40.960885Z","shell.execute_reply":"2025-09-02T18:20:41.180750Z"}}
# 19) Dataset split histogram
n_train=len(train_p); n_val=len(val_p); n_test=len(test_p)
total=n_train+n_val+n_test
counts=[n_train,n_val,n_test]; labels=['Train','Validation','Test']
perc=[c/total*100 for c in counts]
plt.figure(figsize=(8,6))
bars=plt.bar(labels,counts,color=['skyblue','orange','green'])
for i,b in enumerate(bars):
    plt.text(b.get_x()+b.get_width()/2, b.get_height()+1, f"{counts[i]} ({perc[i]:.1f}%)", ha='center', va='bottom')
plt.title("Distribution of patients"); plt.ylabel("Count"); plt.grid(axis='y', ls='--', alpha=0.5)
plt.tight_layout(); plt.savefig("/kaggle/working/dataset_split_histogram_percent.png"); plt.show()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-09-02T18:20:46.048336Z","iopub.execute_input":"2025-09-02T18:20:46.048887Z","iopub.status.idle":"2025-09-02T18:20:49.244008Z","shell.execute_reply.started":"2025-09-02T18:20:46.048864Z","shell.execute_reply":"2025-09-02T18:20:49.243269Z"}}
# 20) ONNX export of final model
try:
    onnx_model = Improved3DUNet(in_channels=4, out_channels=4, base_filters=16).to(device)
    onnx_model.load_state_dict(torch.load(final_path, map_location=device))
    onnx_model.eval()
    dummy, _ = next(iter(test_loader))
    dummy = dummy.to(device)
    torch.onnx.export(onnx_model, dummy, "/kaggle/working/fl/global_model_final.onnx",
                      export_params=True, opset_version=11, do_constant_folding=True,
                      input_names=['input'], output_names=['output'],
                      dynamic_axes={'input':{0:'batch_size'}, 'output':{0:'batch_size'}})
    print("✅ ONNX saved at /kaggle/working/fl/global_model_final.onnx")
except Exception as e:
    print("ONNX export skipped:", e)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-09-02T18:20:55.454610Z","iopub.execute_input":"2025-09-02T18:20:55.455370Z","iopub.status.idle":"2025-09-02T18:20:55.480310Z","shell.execute_reply.started":"2025-09-02T18:20:55.455345Z","shell.execute_reply":"2025-09-02T18:20:55.479764Z"}}
# 21) Verify blockchain hashes against saved ckpts
def verify_blockchain_models(bc_dir="/kaggle/working/blockchain", fl_dir="/kaggle/working/fl"):
    try:
        with open(os.path.join(bc_dir,"ledger.json"),"r") as f:
            ledger=json.load(f)
        ok=True
        for blk in ledger["chain"]:
            pl=blk.get("payload",{})
            if pl.get("type")=="fl_round":
                ckpt=pl.get("global_model_ckpt"); sha=pl.get("global_model_sha256")
                if ckpt and sha:
                    p=os.path.join(fl_dir, ckpt)
                    if os.path.exists(p):
                        curr=file_sha256(p)
                        if curr!=sha: 
                            print("Hash mismatch", ckpt); ok=False
                enc=pl.get("encrypted_ckpt"); enc_sha=pl.get("encrypted_ckpt_sha256")
                if enc and enc_sha:
                    p=os.path.join(fl_dir, enc)
                    if os.path.exists(p):
                        curr=file_sha256(p)
                        if curr!=enc_sha:
                            print("Encrypted hash mismatch", enc); ok=False
            if pl.get("type")=="final_model":
                ckpt=pl.get("final_model_path"); sha=pl.get("final_model_sha256")
                if ckpt and sha:
                    p=os.path.join(fl_dir, ckpt)
                    if os.path.exists(p) and file_sha256(p)!=sha: ok=False
        print("Blockchain-file verification:", "OK" if ok else "FAILED")
        return ok
    except Exception as e:
        print("Verification error:", e); return False

verify_blockchain_models()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-09-02T18:21:06.014475Z","iopub.execute_input":"2025-09-02T18:21:06.015185Z","iopub.status.idle":"2025-09-02T18:21:06.029422Z","shell.execute_reply.started":"2025-09-02T18:21:06.015161Z","shell.execute_reply":"2025-09-02T18:21:06.028723Z"}}
# 22) Prediction utilities + overlays (also used by Grad-CAM / LIME)
import glob as _glob

PRED_DIR="/kaggle/working/preds"; os.makedirs(PRED_DIR, exist_ok=True)
def get_seg_colors(): return SEG_COLORS
def label_to_color_map(seg_slice, seg_colors=None):
    if seg_colors is None: seg_colors=get_seg_colors()
    h,w=seg_slice.shape; out=np.zeros((h,w,3),dtype=np.uint8)
    for k,clr in seg_colors.items(): out[seg_slice==k]=clr
    return out

def pick_checkpoint(load_best_first=True, override=None):
    if override and os.path.exists(override): return override
    fl_dir="/kaggle/working/fl"
    best=os.path.join(fl_dir,"global_model_best.pth")
    final=os.path.join(fl_dir,"global_model_final.pth")
    if load_best_first and os.path.exists(best): return best
    if os.path.exists(final): return final
    rounds=sorted(_glob.glob(os.path.join(fl_dir,"global_round_*.pth")))
    if len(rounds)>0: return rounds[-1]
    raise FileNotFoundError("No checkpoint found")

def get_infer_model(load_best_first=True, override=None):
    lp=pick_checkpoint(load_best_first, override)
    m=Improved3DUNet(4,4,16).to(device); m.load_state_dict(torch.load(lp, map_location=device)); m.eval()
    return m, lp

def load_patient_volume(pid, data_dir, slice_range=(60,100), use_center_slices=True):
    mods=["flair","t1","t1ce","t2"]; p_path=os.path.join(data_dir,pid)
    # Determine depth from flair
    flair_img=nib.load(os.path.join(p_path,f"{pid}_flair.nii.gz"))
    flair_vol=flair_img.get_fdata().astype(np.float32)
    H,W,D=flair_vol.shape
    if use_center_slices:
        z0=max(0,(D-40)//2); z1=z0+40
    else:
        z0,z1=slice_range
    vols=[]; aff=None
    for mname in mods:
        img=nib.load(os.path.join(p_path,f"{pid}_{mname}.nii.gz")); aff=img.affine
        v=img.get_fdata().astype(np.float32)[:,:,z0:z1]
        mu,sd=np.mean(v), np.std(v)+1e-6
        v=(v-mu)/sd; vols.append(np.nan_to_num(v))
    x=np.stack(vols,axis=0)[None,...]  # [1,4,H,W,Dsub]
    seg_path=os.path.join(p_path,f"{pid}_seg.nii.gz")
    seg=None
    if os.path.exists(seg_path):
        s=nib.load(seg_path).get_fdata().astype(np.uint8)[:,:,z0:z1]
        s[s==4]=3; seg=s
    return x, seg, aff, (z0,z1), (H,W,D)

def save_pred_nifti(pid, pred_3d, affine, original_depth, slice_range, out_dir=PRED_DIR):
    z0,z1=slice_range; H,W,Dsub=pred_3d.shape
    full=np.zeros((H,W,original_depth),dtype=np.uint8)
    full[:,:,z0:z1]=pred_3d
    outp=os.path.join(out_dir, f"{pid}_pred_seg.nii.gz")
    nib.save(nib.Nifti1Image(full, affine), outp); return outp

def show_prediction_overlays(pid, data_dir, pred_vol, gt_vol, slice_range, z_global=None):
    p_path=os.path.join(data_dir,pid); z0,z1=slice_range
    if z_global is None: z_global=(z0+z1)//2
    z_local=z_global-z0
    flair=nib.load(os.path.join(p_path,f"{pid}_flair.nii.gz")).get_fdata()
    flair_slice=flair[:,:,z_global]
    pred_color=label_to_color_map(pred_vol[:,:,z_local])
    gt_color=label_to_color_map(gt_vol[:,:,z_local]) if gt_vol is not None else None
    plt.figure(figsize=(14,4))
    plt.subplot(1,3,1); plt.imshow(flair_slice,cmap='gray'); plt.title(f"{pid} | FLAIR z={z_global}"); plt.axis('off')
    plt.subplot(1,3,2); plt.imshow(pred_color); plt.title("Prediction (color)"); plt.axis('off')
    if gt_color is not None:
        plt.subplot(1,3,3); plt.imshow(gt_color); plt.title("Ground Truth (color)"); plt.axis('off')
    else:
        plt.subplot(1,3,3); plt.imshow(flair_slice,cmap='gray'); plt.title("No GT"); plt.axis('off')
    plt.tight_layout(); plt.show()
    return z_global

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-09-02T18:26:16.316880Z","iopub.execute_input":"2025-09-02T18:26:16.317750Z","iopub.status.idle":"2025-09-02T18:26:17.686661Z","shell.execute_reply.started":"2025-09-02T18:26:16.317724Z","shell.execute_reply":"2025-09-02T18:26:17.685924Z"}}
# %% [code]
# 23) Improved Prediction: auto tumor slice + transparent overlays + debug prints

SELECT_MODE = "random"   # "test_first" | "random" | "specific"
SPECIFIC_PID = None      # e.g., "BraTS20_Training_177"
OVERRIDE_CKPT = None     # e.g., "/kaggle/working/fl/global_round_2.pth"

def pick_patient_id(select_mode, specific_pid):
    if select_mode == "test_first":
        if len(test_p) > 0: return test_p[0]
        elif len(train_p) > 0: return train_p[0]
        else: raise RuntimeError("No patients available")
    elif select_mode == "random":
        pool = test_p if len(test_p) > 0 else train_p
        return random.choice(pool)
    elif select_mode == "specific":
        assert specific_pid is not None, "Set SPECIFIC_PID when SELECT_MODE='specific'"
        return specific_pid
    else:
        raise ValueError(f"Unknown SELECT_MODE: {select_mode}")

def label_to_rgba(seg_slice, alpha=0.6):
    # Background (0) => fully transparent
    h, w = seg_slice.shape
    rgba = np.zeros((h, w, 4), dtype=np.float32)
    for lab, color in SEG_COLORS.items():
        if lab == 0:  # keep transparent
            continue
        mask = (seg_slice == lab)
        r, g, b = np.array(color) / 255.0
        rgba[mask, 0] = r
        rgba[mask, 1] = g
        rgba[mask, 2] = b
        rgba[mask, 3] = alpha
    return rgba

try:
    pid = pick_patient_id(SELECT_MODE, SPECIFIC_PID)
    print("Predicting:", pid)
    model, ckpt = get_infer_model(load_best_first=True, override=OVERRIDE_CKPT)

    # Load sub-volume (uses center-40 if enabled in PREPROC)
    vol, gt, aff, SL, orig_shape = load_patient_volume(
        pid, DATA_DIR,
        slice_range=PREPROC["slice_range"],
        use_center_slices=PREPROC["use_center_slices"]
    )
    z0, z1 = SL
    with torch.no_grad():
        x = torch.tensor(vol, dtype=torch.float32, device=device)
        logits = model(x)
        probs  = torch.softmax(logits, dim=1)
        pred   = torch.argmax(probs, dim=1)[0].cpu().numpy().astype(np.uint8)  # [H,W,D_sub]

    # Debug: count non-zero voxels in sub-volume
    pred_nz = int((pred > 0).sum())
    gt_nz   = int((gt > 0).sum()) if gt is not None else None
    print(f"Sub-volume nonzero voxels -> Pred: {pred_nz}" + (f", GT: {gt_nz}" if gt is not None else ""))

    # Pick a slice with tumor if possible (prefer GT, else prediction, else middle)
    if gt is not None and (gt > 0).any():
        counts = (gt > 0).sum(axis=(0, 1))  # per-slice tumor voxels
        z_local = int(np.argmax(counts))
        reason = "GT"
    else:
        counts = (pred > 0).sum(axis=(0, 1))
        if counts.max() > 0:
            z_local = int(np.argmax(counts))
            reason = "Prediction"
        else:
            z_local = (pred.shape[2] // 2)
            reason = "Middle (no tumor found)"
    z_global = z0 + z_local
    print(f"Chosen slice: z_global={z_global} (reason: {reason}) | labels@slice -> pred:{np.unique(pred[:,:,z_local])}" +
          (f", gt:{np.unique(gt[:,:,z_local])}" if gt is not None else ""))

    # Save NIfTI embedded into full depth
    outp = save_pred_nifti(pid, pred, aff, original_depth=orig_shape[2], slice_range=SL, out_dir=PRED_DIR)
    print("Saved prediction:", outp)

    # Prepare overlays with transparent background
    flair_full = nib.load(os.path.join(DATA_DIR, pid, f"{pid}_flair.nii.gz")).get_fdata()
    flair_slice = flair_full[:, :, z_global]
    pred_rgba = label_to_rgba(pred[:, :, z_local], alpha=0.55)
    gt_rgba   = label_to_rgba(gt[:, :, z_local],   alpha=0.55) if gt is not None else None

    # Plot: FLAIR, FLAIR+Pred, FLAIR+GT (if available)
    plt.figure(figsize=(15, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(flair_slice, cmap="gray")
    plt.title(f"{pid} | FLAIR z={z_global}")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(flair_slice, cmap="gray")
    plt.imshow(pred_rgba)  # RGBA handles transparency
    plt.title("Prediction overlay")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    if gt_rgba is not None:
        plt.imshow(flair_slice, cmap="gray")
        plt.imshow(gt_rgba)
        plt.title("Ground Truth overlay")
    else:
        plt.imshow(flair_slice, cmap="gray")
        plt.title("No GT available")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    # Expose selection for Grad-CAM/LIME
    SELECTED_PID = pid
    SELECTED_Z_GLOBAL = z_global
    SELECTED_SLICE_RANGE = SL
    SELECTED_MODEL_PATH = ckpt
    SELECTED_DATA_DIR = DATA_DIR

except Exception as e:
    print("Prediction error:", e)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-09-02T18:26:26.528187Z","iopub.execute_input":"2025-09-02T18:26:26.528924Z","iopub.status.idle":"2025-09-02T18:26:28.893205Z","shell.execute_reply.started":"2025-09-02T18:26:26.528898Z","shell.execute_reply":"2025-09-02T18:26:28.892514Z"}}
# 24) Grad-CAM (slice-level heatmap overlay)
def gradcam_3d(model, x, target_class=3, layer_attr="dec1", slice_index=None, use_pred_mask=True):
    model.eval()
    acts,grads=[],[]
    layer=getattr(model, layer_attr)
    def fwd_hook(m,inp,out): acts.append(out.detach())
    def bwd_hook(m,gin,gout): grads.append(gout[0].detach())
    h1=layer.register_forward_hook(fwd_hook)
    try:   h2=layer.register_full_backward_hook(bwd_hook)
    except: h2=layer.register_backward_hook(bwd_hook)

    x=x.clone().requires_grad_(True)
    logits=model(x); probs=torch.softmax(logits,dim=1)
    _,C,H,W,D=logits.shape
    if slice_index is None: slice_index=D//2
    plabels=torch.argmax(probs,dim=1)
    sel=(plabels[0,:,:,slice_index]==target_class)
    target_map=logits[0,target_class,:,:,slice_index]
    target=target_map[sel].mean() if (use_pred_mask and sel.sum()>0) else target_map.mean()

    model.zero_grad(); target.backward(retain_graph=True)
    A=acts[-1]; G=grads[-1]
    w=G.mean(dim=(2,3,4), keepdim=True)
    cam=(w*A).sum(dim=1)[0]; cam=torch.relu(cam)
    cam=(cam-cam.min())/(cam.max()-cam.min()+1e-8)
    h1.remove(); h2.remove()
    return cam.detach().cpu().numpy()

try:
    pid_gc=SELECTED_PID if 'SELECTED_PID' in globals() else random.choice(test_p)
    SL=SELECTED_SLICE_RANGE if 'SELECTED_SLICE_RANGE' in globals() else (60,100)
    data_dir_gc=SELECTED_DATA_DIR if 'SELECTED_DATA_DIR' in globals() else DATA_DIR
    z_global=SELECTED_Z_GLOBAL if 'SELECTED_Z_GLOBAL' in globals() else None

    model_gc, lp = get_infer_model(load_best_first=True, override=None)
    vol_gc, gt_gc, _aff, _SL, _orig = load_patient_volume(pid_gc, data_dir_gc, slice_range=PREPROC["slice_range"],
                                                          use_center_slices=PREPROC["use_center_slices"])
    x_gc=torch.tensor(vol_gc,dtype=torch.float32,device=device)
    z0,z1=SL; z_global=(z0+z1)//2 if z_global is None else z_global; z_local=z_global-z0

    with torch.no_grad():
        logits_gc=model_gc(x_gc)
        pred_gc=torch.argmax(torch.softmax(logits_gc, dim=1),dim=1)[0].cpu().numpy().astype(np.uint8)

    cam3d=gradcam_3d(model_gc, x_gc, target_class=3, layer_attr="dec1", slice_index=z_local, use_pred_mask=True)
    cam_slice=cam3d[:,:,z_local]

    flair_full=nib.load(os.path.join(data_dir_gc, pid_gc, f"{pid_gc}_flair.nii.gz")).get_fdata()
    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1); plt.imshow(flair_full[:,:,z_global],cmap='gray'); plt.title(f"{pid_gc} | FLAIR z={z_global}"); plt.axis('off')
    plt.subplot(1,3,2); plt.imshow(flair_full[:,:,z_global],cmap='gray'); plt.imshow(cam_slice,cmap='jet',alpha=0.5)
    plt.title("Grad-CAM (class 3)"); plt.axis('off')
    plt.subplot(1,3,3); plt.imshow(label_to_color(pred_gc[:,:,z_local])); plt.title("Prediction (color)"); plt.axis('off')
    plt.tight_layout(); plt.show()
except Exception as e:
    print("Grad-CAM error:", e)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-09-02T18:21:31.116154Z","iopub.execute_input":"2025-09-02T18:21:31.116871Z","iopub.status.idle":"2025-09-02T18:22:49.804876Z","shell.execute_reply.started":"2025-09-02T18:21:31.116830Z","shell.execute_reply":"2025-09-02T18:22:49.804129Z"}}
# 25) LIME (2D slice-level)
def _ensure_lime():
    try:
        from lime import lime_image
        from skimage.segmentation import mark_boundaries
        return lime_image, mark_boundaries
    except Exception:
        subprocess.run([sys.executable,"-m","pip","install","-q","lime","scikit-image"], check=False)
        from lime import lime_image
        from skimage.segmentation import mark_boundaries
        return lime_image, mark_boundaries

lime_image, mark_boundaries = _ensure_lime()

def build_slice_rgb(flair, t1ce, t2):
    def to01(a):
        mn,mx=np.min(a),np.max(a)
        return (a-mn)/(mx-mn+1e-8), (mn,mx)
    r,(rmin,rmax)=to01(flair); g,(gmin,gmax)=to01(t1ce); b,(bmin,bmax)=to01(t2)
    rgb=np.stack([r,g,b],axis=-1).astype(np.float32)
    rng={'r':(rmin,rmax),'g':(gmin,gmax),'b':(bmin,bmax)}
    return rgb, rng

def lime_explain_slice(pid, data_dir, model, slice_range=(60,100), target_class=3, z_global=None, num_samples=600):
    vol, gt, aff, SL, _orig = load_patient_volume(pid, data_dir, slice_range=slice_range, use_center_slices=True)
    base=vol.copy()[0]  # [4,H,W,Dsub]
    z0,z1=SL
    if z_global is None: z_global=(z0+z1)//2
    z_local=z_global-z0
    flair=base[0,:,:,z_local]; t1ce=base[2,:,:,z_local]; t2=base[3,:,:,z_local]
    rgb,rng=build_slice_rgb(flair,t1ce,t2)

    def predict_fn(batch_rgb):
        outs=[]
        with torch.no_grad():
            for img in batch_rgb:
                r=img[:,:,0]*(rng['r'][1]-rng['r'][0])+rng['r'][0]
                g=img[:,:,1]*(rng['g'][1]-rng['g'][0])+rng['g'][0]
                b=img[:,:,2]*(rng['b'][1]-rng['b'][0])+rng['b'][0]
                vol_mod=base.copy()
                vol_mod[0,:,:,z_local]=r; vol_mod[2,:,:,z_local]=g; vol_mod[3,:,:,z_local]=b
                x=torch.tensor(vol_mod[None,...],dtype=torch.float32,device=device)
                logits=model(x); probs=torch.softmax(logits,dim=1)[0,target_class,:,:,z_local].detach().cpu().numpy()
                p=float(np.mean(probs)); outs.append([1.0-p,p])
        return np.array(outs,dtype=np.float32)

    base_probs=predict_fn(rgb[None,...])[0]
    print(f"LIME base slice probs (class {target_class}): no={base_probs[0]:.4f}, yes={base_probs[1]:.4f}")

    explainer=lime_image.LimeImageExplainer(random_state=42)
    explanation=explainer.explain_instance(rgb, predict_fn, top_labels=2, hide_color=0, num_samples=num_samples, segmentation_fn=None)
    label_sel=1 if 1 in explanation.top_labels else explanation.top_labels[0]
    temp, mask = explanation.get_image_and_mask(label=label_sel, positive_only=True, hide_rest=False, num_features=10, min_weight=0.0)

    flair_full=nib.load(os.path.join(data_dir, pid, f"{pid}_flair.nii.gz")).get_fdata()
    fig,ax=plt.subplots(1,3,figsize=(14,4))
    ax[0].imshow(rgb); ax[0].set_title(f"{pid} RGB z={z_global}"); ax[0].axis('off')
    ax[1].imshow(mark_boundaries(rgb, mask)); ax[1].set_title("LIME positive"); ax[1].axis('off')
    ax[2].imshow(flair_full[:,:,z_global],cmap='gray'); ax[2].contour(mask,colors='lime',linewidths=1.2,levels=[0.5])
    ax[2].set_title("FLAIR + LIME regions"); ax[2].axis('off')
    plt.tight_layout(); plt.show()

try:
    model_lime, _lp = get_infer_model(load_best_first=True)
    lime_explain_slice(SELECTED_PID if 'SELECTED_PID' in globals() else random.choice(test_p),
                       DATA_DIR, model_lime, slice_range=PREPROC["slice_range"], target_class=3,
                       z_global=SELECTED_Z_GLOBAL if 'SELECTED_Z_GLOBAL' in globals() else None, num_samples=400)
except Exception as e:
    print("LIME error:", e)
