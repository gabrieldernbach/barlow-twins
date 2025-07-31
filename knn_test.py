# This file is a copy from
# https://github.com/yaohungt/Barlow-Twins-HSIC/blob/main/main.py

import torch
from tqdm import tqdm

def test(net, db):
    temperature = 0.5
    k = 200
    c = 10

    net.eval()
    total_top1, total_top5, total_num = 0.0, 0.0, 0
    
    # Memory-efficient feature extraction
    feature_bank = []
    target_bank = []
    
    with torch.no_grad():
        # Generate feature bank and target bank with memory optimization
        for data, target in tqdm(db.memory, desc='Feature extracting'):
            feature, _ = net(data.cuda(non_blocking=True))
            # Detach and move to CPU to save GPU memory
            feature_bank.append(feature.detach().cpu())
            target_bank.append(target)
            
        # Concatenate on CPU first, then move to GPU
        feature_bank = torch.cat(feature_bank, dim=0)
        feature_labels = torch.cat(target_bank, dim=0)
        
        # Move to GPU only when needed and transpose
        feature_bank = feature_bank.cuda().t().contiguous()
        feature_labels = feature_labels.cuda().contiguous()
        
        # Process test data in chunks to avoid memory issues
        test_bar = tqdm(db.test)
        for data, target in test_bar:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            feature, _ = net(data)

            total_num += data.size(0)
            
            # Compute cosine similarity more efficiently
            sim_matrix = torch.mm(feature, feature_bank)
            
            # Get top-k similarities and indices
            sim_weight, sim_indices = sim_matrix.topk(k=k, dim=-1)
            
            # Gather corresponding labels
            sim_labels = torch.gather(feature_labels.expand(data.size(0), -1), dim=-1, index=sim_indices)
            
            # Apply temperature scaling
            sim_weight = (sim_weight / temperature).exp()

            # More efficient one-hot encoding and weighted scoring
            batch_size = data.size(0)
            one_hot_label = torch.zeros(batch_size, k, c, device=sim_labels.device)
            one_hot_label.scatter_(dim=-1, index=sim_labels.unsqueeze(-1), value=1.0)
            
            # Weighted score computation
            pred_scores = torch.sum(one_hot_label * sim_weight.unsqueeze(-1), dim=1)

            pred_labels = pred_scores.argsort(dim=-1, descending=True)
            total_top1 += torch.sum((pred_labels[:, :1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            total_top5 += torch.sum((pred_labels[:, :5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            test_bar.set_description('Test Epoch: Acc@1:{:.2f}% Acc@5:{:.2f}%'
                                     .format(total_top1 / total_num * 100, total_top5 / total_num * 100))

    return total_top1 / total_num * 100, total_top5 / total_num * 100
