import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from sklearn import metrics


class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(LinkPredictor, self).__init__()

        self.lins = torch.nn.ModuleList()
        m = torch.nn.Bilinear(25, 26, 32) 
        self.lins.append(m)

        self.lins.append(torch.nn.Linear(32 + 1, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x_i, x_j, x_ij):
        x = self.lins[0](x_i, x_j)
        x = torch.cat([x, x_ij], dim=1)
        for lin in self.lins[1:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)


def train(predictor, x, optimizer, batch_size, device):
    predictor.train()
    total_loss = total_examples = 0
    for src_x, dst_x, y, src_dst_x in iter(x):
        optimizer.zero_grad()

        src_x_pos, dst_x_pos, src_dst_x_pos = [], [], []
        src_x_neg, dst_x_neg, src_dst_x_neg = [], [], []

        for(idx, label) in enumerate(y):
            #print(label)
            if label == 1:
                src_x_pos.append(src_x[idx].numpy())
                dst_x_pos.append(dst_x[idx].numpy())
                src_dst_x_pos.append(src_dst_x[idx].numpy())
            else:
                src_x_neg.append(src_x[idx].numpy())
                dst_x_neg.append(dst_x[idx].numpy())
                src_dst_x_neg.append(src_dst_x[idx].numpy())
            
        
        src_x_pos = np.array(src_x_pos)
        src_x_pos = torch.from_numpy(src_x_pos).float().to(device)
        dst_x_pos = np.array(dst_x_pos)
        dst_x_pos = torch.from_numpy(dst_x_pos).float().to(device)
        src_dst_x_pos = np.array(src_dst_x_pos)
        src_dst_x_pos = torch.from_numpy(src_dst_x_pos).float().to(device)


        src_x_neg = np.array(src_x_neg)
        src_x_neg = torch.from_numpy(src_x_neg).float().to(device)
        dst_x_neg = np.array(dst_x_neg)
        dst_x_neg = torch.from_numpy(dst_x_neg).float().to(device)
        src_dst_x_neg = np.array(src_dst_x_neg)
        src_dst_x_neg = torch.from_numpy(src_dst_x_neg).float().to(device)
        #print(src_x_pos.shape,src_x_neg.shape)
        pos_out = predictor(src_x_pos, dst_x_pos, src_dst_x_pos)
 
        pos_loss = -torch.log(pos_out + 1e-15).mean()
        
        neg_out = predictor(src_x_neg, dst_x_neg, src_dst_x_neg)
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()

        loss = pos_loss + neg_loss
        loss.backward()
        optimizer.step()

        num_examples = pos_out.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples

    return total_loss / total_examples


@torch.no_grad()
def test(predictor, val_dataloader, batch_size, device):
    predictor.eval()
    preds = []
    for src_x, dst_x, y, src_dst_x  in iter(val_dataloader):

        src_x =  src_x.float().to(device)
        dst_x = dst_x.float().to(device)
        src_dst_x = src_dst_x.float().to(device)
        preds += [predictor(src_x, dst_x, src_dst_x).squeeze().cpu()]

    preds = torch.cat(preds, dim=0).numpy()

    top_5_back, lines_to_tdw = link_pred_metrics(preds)

    return top_5_back, lines_to_tdw


def load_data(split):
    src_x, dst_x, y, src_dst_x = [], [], [], []
    with open("data/" + split,"r") as f:
        lines = f.readlines()
        for line in lines:
            tmp = line.strip().split("\t")
            reward = float(tmp[6])
            a_x_array = tmp[7:32]
            a_x_array = np.array([float(x) for x in a_x_array])
            src_x.append(a_x_array)

            b_x_array = tmp[33:]
            b_x_array = np.array([float(x) for x in b_x_array])
            dst_x.append(b_x_array)

            if reward > 0.0:
                y.append(1)
            else:
                y.append(0)

            ab_intimacy = float(tmp[32])
            src_dst_x.append(np.array([ab_intimacy]))

    src_x = np.array(src_x)
    dst_x = np.array(dst_x)
    y = np.array(y)
    src_dst_x = np.array(src_dst_x)

    src_x = torch.from_numpy(src_x)
    dst_x = torch.from_numpy(dst_x)
    y = torch.from_numpy(y)
    src_dst_x = torch.from_numpy(src_dst_x)
    print(src_x.shape, dst_x.shape, y.shape, src_dst_x.shape)

    return src_x, dst_x, y, src_dst_x   


def link_pred_metrics(scores):

    # (往期) 上线的召回列表，src：左端点ID，dst: 右端点ID。
    recall_candidates = {}
    # 实际的成功召回列表(即左端点点击了右端点，且右端点回流)，如果一个左端没有点击，或没有右端点被召回，则字典中没有左端点src这个key
    real_callbacks = {}

    print(scores.shape)
    labels = []

    lines_to_tdw = []

    with open("data/val", "r") as f:
        lines = f.readlines()
        for (i, line) in enumerate(lines):
            tmp = line.strip().split("\t")
            openid = tmp[0]
            fopenid = tmp[1]
            score = scores[i]
            reward = float(tmp[6])
            if reward > 0.0:
                labels.append(1)
            else:
                labels.append(0)
            lines_to_tdw.append(openid + "\t" + fopenid + "\t" + str(score) + "\n")

            if openid not in recall_candidates:
                recall_candidates[openid] = []
                tmp_list = recall_candidates[openid]
                tmp_list.append([fopenid, score])
                recall_candidates[openid] = tmp_list
                
            else:
                tmp_list = recall_candidates[openid]
                tmp_list.append([fopenid, score])
                recall_candidates[openid] = tmp_list

            if reward > 0.0:
                if openid not in real_callbacks:
                    real_callbacks[openid] = [fopenid]
                else:
                    back_friends = real_callbacks[openid]
                    back_friends.append(fopenid)
                    real_callbacks[openid] = back_friends

    sorted_recall_candidates = {}
    for openid in recall_candidates:
        rec_list = recall_candidates[openid]
        rec_list = np.array(rec_list)
        sorted_rec_list = rec_list[rec_list[:,1].argsort()[::-1]]
        sorted_recall_candidates[openid] = sorted_rec_list[:, 0]

    ranks = []
    hits = []
    for i in range(10):
        hits.append([])

    for src in real_callbacks:
        
        algo_ranks = sorted_recall_candidates[src]
        #print(src, real_callbacks[src], algo_ranks)
        for real_dst in real_callbacks[src]:
            rank = 0
            for (i, e) in enumerate(algo_ranks):
                if e == real_dst:
                    rank = i + 1
                    #print(rank)
                    ranks.append(rank)
                    break
            for hits_level in range(10):
                if rank -1 <= hits_level:
                    hits[hits_level].append(1.0)
                else:
                    hits[hits_level].append(0.0)
    # 平均排名
    mean_rank = np.mean(ranks)
    print("Mean rank: ", mean_rank)

    # 排名倒数的平均
    mrr = np.mean(1./np.array(ranks))
    print("Mean reciprocal rank: ", mrr)

    # 前1，前3，前5的命中率：
    for i in [0,2,4,9]:
        print('Hits @{0}: {1}'.format(i+1, np.mean(hits[i])))

    top_5_back = np.sum(hits[4])
    print("top 5 back number: ", top_5_back)
    top_10_back = np.sum(hits[9])
    print("top 10 back number: ", top_10_back)

    fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    print("AUC: ", auc)

    return top_5_back, lines_to_tdw

def main():
    parser = argparse.ArgumentParser(description='Blinear Link Prediction')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--eval_steps', type=int, default=1)
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--output_model_path', type=str, default='results/bilinear_model.pt')

    args = parser.parse_args()
    print(args)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    src_x, dst_x, y, src_dst_x = load_data("train")
    train_data = [(src_x[idx], dst_x[idx], y[idx], src_dst_x[idx]) for idx in range(len(src_x))]
    train_dataloader = torch.utils.data.DataLoader(train_data, 512, shuffle=True, num_workers=4)
    print(len(train_dataloader))
    src_x, dst_x, y, src_dst_x = load_data("val")

    val_data = [(src_x[idx], dst_x[idx], y[idx], src_dst_x[idx]) for idx in range(len(src_x))]
    val_dataloader = torch.utils.data.DataLoader(val_data, 256, shuffle=False, num_workers=4)    

    predictor = LinkPredictor(25, args.hidden_channels, 1,
                              args.num_layers, args.dropout).to(device)

    for run in range(args.runs):
        predictor.reset_parameters()
        optimizer = torch.optim.Adam(predictor.parameters(), lr=args.lr)

        best_top5 = 0

        for epoch in range(1, 1 + args.epochs):
            loss = train(predictor, train_dataloader, optimizer,
                         args.batch_size, device)
            print(f'Run: {run + 1:02d}, Epoch: {epoch:02d}, Loss: {loss:.4f}')

            top_5_back, lines_to_tdw = test(predictor, val_dataloader, args.batch_size, device)
            if top_5_back > best_top5:
                best_top5 = top_5_back
                print(f'saving best model with top 5 back: {best_top5} at {epoch} epoch!')  
                torch.save(predictor, args.output_model_path)
                with open("results/scores_bilinear_val", "w") as f:
                    f.writelines(lines_to_tdw)

if __name__ == "__main__":
    main()