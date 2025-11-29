import argparse
import torch
import torch.nn.functional as F
from data.dataset import get_data
from models.model import AGFNodeClassifier
import time
import numpy as np

def train(model, optimizer, data, device, reg_weight):
    model.train()
    optimizer.zero_grad()
    
    x = data.x.to(device)
    y = data.y.to(device)
    train_mask = data.train_mask.to(device)
    
    logits, ortho_loss = model(x)
    
    loss = F.cross_entropy(logits[train_mask], y[train_mask])
    
    if isinstance(ortho_loss, torch.Tensor):
        total_loss = loss + reg_weight * ortho_loss
    else:
        total_loss = loss
        
    total_loss.backward()
    optimizer.step()
    
    return loss.item(), ortho_loss.item() if isinstance(ortho_loss, torch.Tensor) else 0

@torch.no_grad()
def test(model, data, device):
    model.eval()
    x = data.x.to(device)
    y = data.y.to(device)
    
    logits, _ = model(x)
    pred = logits.argmax(dim=-1)
    
    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        mask = mask.to(device)
        accs.append(int((pred[mask] == y[mask]).sum()) / int(mask.sum()))
    return accs

def main():
    parser = argparse.ArgumentParser(description='AGF Node Classification')
    parser.add_argument('--dataset', type=str, default='Cora', choices=['Cora', 'Citeseer', 'Deezer'])
    parser.add_argument('--d_model', type=int, default=64)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--reg_weight', type=float, default=0.1, help='Weight for orthogonality loss')
    parser.add_argument('--attn_type', type=str, default='svd', choices=['softmax', 'svd', 'hybrid'])
    parser.add_argument('--poly_type', type=str, default='jacobi')
    parser.add_argument('--K', type=int, default=3)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    print(args)
    
    device = torch.device(args.device)
    
    # Load data
    try:
        dataset, data = get_data(args.dataset)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    print(f"Dataset: {args.dataset}")
    print(f"Num Nodes: {data.num_nodes}")
    print(f"Num Features: {dataset.num_features}")
    print(f"Num Classes: {dataset.num_classes}")
    
    # Check masks
    if not hasattr(data, 'train_mask'):
        print("Generating random splits...")
        # Simple random split if not present
        indices = torch.randperm(data.num_nodes)
        train_len = int(0.6 * data.num_nodes)
        val_len = int(0.2 * data.num_nodes)
        
        data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        data.train_mask[indices[:train_len]] = True
        
        data.val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        data.val_mask[indices[train_len:train_len+val_len]] = True
        
        data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        data.test_mask[indices[train_len+val_len:]] = True
        
    model = AGFNodeClassifier(
        num_features=dataset.num_features,
        num_classes=dataset.num_classes,
        d_model=args.d_model,
        n_heads=args.n_heads,
        num_layers=args.num_layers,
        attn_type=args.attn_type,
        poly_type=args.poly_type,
        K=args.K
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    best_val_acc = 0
    test_acc = 0
    
    for epoch in range(1, args.epochs + 1):
        loss, ortho_loss = train(model, optimizer, data, device, args.reg_weight)
        train_acc, val_acc, tmp_test_acc = test(model, data, device)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            test_acc = tmp_test_acc
            
        if epoch % 10 == 0:
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Ortho: {ortho_loss:.4f}, '
                  f'Train: {train_acc:.4f}, Val: {best_val_acc:.4f}, Test: {test_acc:.4f}')
            
    print(f'Final Test Acc: {test_acc:.4f}')

if __name__ == "__main__":
    main()
