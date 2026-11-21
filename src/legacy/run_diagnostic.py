import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

DATA_DIR = "data/processed/SEED"
NUM_SUBJECTS = 15
NUM_CLASSES = 3
BATCH_SIZE = 256
LATENT_DIM = 32
SEED = 42

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_subject(subject_id):
    X_list, y_list = [], []
    for session in [1, 2, 3]:
        path = os.path.join(DATA_DIR, str(session), f"subject{subject_id}.pth")
        data = torch.load(path, weights_only=True)
        session_X = torch.stack(data["data"]).float()
        session_y = torch.tensor(data["labels"], dtype=torch.long)
        X_list.append(session_X)
        y_list.append(session_y)
    return torch.cat(X_list, dim=0), torch.cat(y_list, dim=0)

def standardize(source_X, target_X):
    mean = source_X.mean(dim=0, keepdim=True)
    std = source_X.std(dim=0, keepdim=True)
    std = torch.clamp(std, min=1e-6)
    return (source_X - mean) / std, (target_X - mean) / std

# Multi-Kernel MMD
def multi_kernel_mmd_loss(source, target, sigmas=(1.0, 2.0, 5.0, 10.0, 20.0, 40.0)):
    """
    Multi-Kernel Maximum Mean Discrepancy with a mixture of RBF bandwidths.
    Avoids kernel saturation and vanishing gradients in high/medium dimensions.
    """
    s_ss = (source.unsqueeze(1) - source.unsqueeze(0)).pow(2).sum(dim=2)
    s_tt = (target.unsqueeze(1) - target.unsqueeze(0)).pow(2).sum(dim=2)
    s_st = (source.unsqueeze(1) - target.unsqueeze(0)).pow(2).sum(dim=2)

    k_ss = torch.zeros_like(s_ss)
    k_tt = torch.zeros_like(s_tt)
    k_st = torch.zeros_like(s_st)

    for sigma in sigmas:
        gamma = 1.0 / (2.0 * (sigma ** 2))
        k_ss = k_ss + torch.exp(-gamma * s_ss)
        k_tt = k_tt + torch.exp(-gamma * s_tt)
        k_st = k_st + torch.exp(-gamma * s_st)

    k_ss = k_ss / len(sigmas)
    k_tt = k_tt / len(sigmas)
    k_st = k_st / len(sigmas)

    return k_ss.mean() + k_tt.mean() - 2.0 * k_st.mean()

from src.legacy.scvae import SCVAE, reconstruction_loss, kl_loss, supervised_contrastive_loss, classification_loss, mmd_loss

def run_diagnostic_experiment(test_subject=2, epochs=20, lr=1e-3, config_name="default", use_mk_mmd=False, loss_weights=None):
    set_seed(SEED)
    print("=" * 75)
    print(f"DIAGNOSTIC RUN: Config = {config_name} | Target Subject = {test_subject} | Device = {DEVICE}")
    print("=" * 75)

    source_subjects = [s for s in range(1, NUM_SUBJECTS + 1) if s != test_subject]
    source_X_list, source_y_list = [], []
    for s in source_subjects:
        X, y = load_subject(s)
        source_X_list.append(X)
        source_y_list.append(y)

    source_X = torch.cat(source_X_list, dim=0)
    source_y = torch.cat(source_y_list, dim=0)
    target_X, target_y = load_subject(test_subject)

    source_X, target_X = standardize(source_X, target_X)

    source_dataset = TensorDataset(source_X, source_y)
    target_dataset = TensorDataset(target_X)

    source_loader = DataLoader(source_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    target_loader = DataLoader(target_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    target_eval_loader = DataLoader(TensorDataset(target_X, target_y), batch_size=BATCH_SIZE, shuffle=False)
    source_eval_loader = DataLoader(source_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = SCVAE(channels=62, feature_dim=5, latent_dim=LATENT_DIM, num_classes=NUM_CLASSES).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if loss_weights is None:
        loss_weights = {"recon": 1.0, "kl": 1.0, "cls": 1.0, "supcon": 1.0, "mmd": 1.0}

    print(f"Loss weights: {loss_weights}")
    print(f"Use MK-MMD: {use_mk_mmd}")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss_sum = 0.0
        recon_sum = 0.0
        kl_sum = 0.0
        cls_sum = 0.0
        supcon_sum = 0.0
        mmd_sum = 0.0
        num_batches = 0

        target_iterator = iter(target_loader)

        for source_batch_X, source_batch_y in source_loader:
            try:
                target_batch_X = next(target_iterator)[0]
            except StopIteration:
                target_iterator = iter(target_loader)
                target_batch_X = next(target_iterator)[0]

            source_batch_X = source_batch_X.to(DEVICE)
            source_batch_y = source_batch_y.to(DEVICE)
            target_batch_X = target_batch_X.to(DEVICE)

            optimizer.zero_grad()

            # Forward source and target
            source_rec, source_mu, source_logvar, source_z, source_logits = model(source_batch_X)
            target_rec, target_mu, target_logvar, target_z, target_logits = model(target_batch_X)

            # Multi-view contrastive views
            source_views = model.get_contrastive_views(source_mu) # [B, 2, 32]

            # Losses
            l_recon = (reconstruction_loss(source_rec, source_batch_X) + reconstruction_loss(target_rec, target_batch_X)) / 2.0
            l_kl = (kl_loss(source_mu, source_logvar) + kl_loss(target_mu, target_logvar)) / 2.0
            l_cls = classification_loss(source_logits, source_batch_y)
            l_supcon = supervised_contrastive_loss(source_views, source_batch_y)

            if use_mk_mmd:
                l_mmd = multi_kernel_mmd_loss(source_mu, target_mu)
            else:
                l_mmd = mmd_loss(source_mu, target_mu, sigma=1.0)

            total_loss = (
                loss_weights["recon"] * l_recon
                + loss_weights["kl"] * l_kl
                + loss_weights["cls"] * l_cls
                + loss_weights["supcon"] * l_supcon
                + loss_weights["mmd"] * l_mmd
            )

            total_loss.backward()
            optimizer.step()

            total_loss_sum += total_loss.item()
            recon_sum += l_recon.item()
            kl_sum += l_kl.item()
            cls_sum += l_cls.item()
            supcon_sum += l_supcon.item()
            mmd_sum += l_mmd.item()
            num_batches += 1

        # Evaluate target at this epoch
        model.eval()
        target_preds, target_trues = [], []
        target_logits_list = []
        with torch.no_grad():
            for X_b, y_b in target_eval_loader:
                X_b = X_b.to(DEVICE)
                _, _, _, _, logits = model(X_b)
                target_preds.extend(logits.argmax(dim=1).cpu().numpy())
                target_trues.extend(y_b.numpy())
                target_logits_list.append(logits.cpu())

        target_logits_all = torch.cat(target_logits_list, dim=0).numpy()
        tgt_acc = accuracy_score(target_trues, target_preds)
        tgt_f1 = f1_score(target_trues, target_preds, average="macro")

        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            print(
                f"Epoch {epoch:02d}/{epochs:02d} | "
                f"Loss: {total_loss_sum/num_batches:.4f} | "
                f"Cls: {cls_sum/num_batches:.4f} | "
                f"SupCon: {supcon_sum/num_batches:.4f} | "
                f"MMD: {mmd_sum/num_batches:.4f} | "
                f"Target Acc: {tgt_acc:.4f} | F1: {tgt_f1:.4f}"
            )

    # Final evaluation
    model.eval()
    source_preds, source_trues = [], []
    with torch.no_grad():
        for X_b, y_b in source_eval_loader:
            X_b = X_b.to(DEVICE)
            _, _, _, _, logits = model(X_b)
            source_preds.extend(logits.argmax(dim=1).cpu().numpy())
            source_trues.extend(y_b.numpy())

    src_acc = accuracy_score(source_trues, source_preds)
    src_f1 = f1_score(source_trues, source_preds, average="macro")
    cm = confusion_matrix(target_trues, target_preds)

    print("\n" + "-" * 60)
    print(f"FINAL RESULTS: {config_name}")
    print(f"Source Acc: {src_acc:.4f} | Source F1: {src_f1:.4f}")
    print(f"Target Acc: {tgt_acc:.4f} | Target F1: {tgt_f1:.4f}")
    print(f"Prediction counts: {np.bincount(target_preds, minlength=NUM_CLASSES)}")
    print(f"Target True counts: {np.bincount(target_trues, minlength=NUM_CLASSES)}")
    print("Confusion Matrix:\n", cm)
    print("Target Logit Mean:", target_logits_all.mean(axis=0))
    print("Target Logit Std :", target_logits_all.std(axis=0))
    print("-" * 60 + "\n")

    return {
        "config": config_name,
        "src_acc": src_acc,
        "src_f1": src_f1,
        "tgt_acc": tgt_acc,
        "tgt_f1": tgt_f1,
        "cm": cm
    }

if __name__ == "__main__":
    # Test on Subject 2 (baseline SVM = 54.55%, MLP = 45.25%)
    # Test 1: Equal weights (all 1.0) with single sigma MMD
    run_diagnostic_experiment(test_subject=2, epochs=20, config_name="Equal_Weights_Single_MMD",
                              use_mk_mmd=False, loss_weights={"recon": 1.0, "kl": 1.0, "cls": 1.0, "supcon": 1.0, "mmd": 1.0})

    # Test 2: Paper weights with Single sigma MMD
    run_diagnostic_experiment(test_subject=2, epochs=20, config_name="Paper_Weights_Single_MMD",
                              use_mk_mmd=False, loss_weights={"recon": 0.001, "kl": 0.0012, "cls": 1.0, "supcon": 0.052, "mmd": 0.1})

    # Test 3: Paper weights with Multi-Kernel MMD
    run_diagnostic_experiment(test_subject=2, epochs=20, config_name="Paper_Weights_MK_MMD",
                              use_mk_mmd=True, loss_weights={"recon": 0.001, "kl": 0.0012, "cls": 1.0, "supcon": 0.052, "mmd": 0.1})
