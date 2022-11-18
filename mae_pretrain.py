

import argparse
import math
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToTensor, Compose, Normalize
from tqdm import tqdm

from model import *
from utils import setup_seed


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

    src_x = torch.from_numpy(src_x).float()
    dst_x = torch.from_numpy(dst_x).float()
    y = torch.from_numpy(y)
    src_dst_x = torch.from_numpy(src_dst_x).float()
    print(src_x.shape, dst_x.shape, y.shape, src_dst_x.shape)

    return src_x, dst_x, y, src_dst_x

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--max_device_batch_size', type=int, default=512)
    parser.add_argument('--base_learning_rate', type=float, default=1.5e-4)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--mask_ratio', type=float, default=1.0/3)
    parser.add_argument('--total_epoch', type=int, default=100)
    parser.add_argument('--warmup_epoch', type=int, default=20)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--model_path', type=str, default='results/edge-mae.pt')


    parser.add_argument('--num_hops', type=int, default=3)
    parser.add_argument('--emb_dim', type=int, default=256)
    parser.add_argument('--encoder_num_layer', type=int, default=2)
    parser.add_argument('--decoder_num_layer', type=int, default=1)
    parser.add_argument('--num_head', type=int, default=3)

    parser.add_argument('--h_dim', type=int, default=25)
    parser.add_argument('--r_dim', type=int, default=1)
    parser.add_argument('--t_dim', type=int, default=26)

    args = parser.parse_args()

    setup_seed(args.seed)

    batch_size = args.batch_size
    load_batch_size = min(args.max_device_batch_size, batch_size)

    assert batch_size % load_batch_size == 0
    steps_per_update = batch_size // load_batch_size


    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    src_x, dst_x, y, src_dst_x = load_data("unlabeled")
    train_data = [(src_x[idx], dst_x[idx], y[idx], src_dst_x[idx]) for idx in range(len(src_x))]
    train_dataloader = torch.utils.data.DataLoader(train_data, 2048, shuffle=True, num_workers=4)

    val_data = [(src_x[idx], dst_x[idx], y[idx], src_dst_x[idx]) for idx in range(len(src_x))]
    val_dataloader = torch.utils.data.DataLoader(val_data, 256, shuffle=False, num_workers=4) 

    #model = MAE_ViT_pretrain(mask_ratio=args.mask_ratio, num_hops=3, emb_dim=args.hidden_channels,h_dim=25,r_dim=1,t_dim=26).to(device)
    model = MAE_E2E(args).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.base_learning_rate * args.batch_size / 256, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    lr_func = lambda epoch: min((epoch + 1) / (args.warmup_epoch + 1e-8), 0.5 * (math.cos(epoch / args.total_epoch * math.pi) + 1))
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_func, verbose=True)

    step_count = 0
    optim.zero_grad()
    for e in range(args.total_epoch):
        model.train()
        losses = []

        for src_x, dst_x, y, src_dst_x in iter(train_dataloader):
            step_count += 1
            src_x = src_x.to(device)
            dst_x = dst_x.to(device)
            src_dst_x = src_dst_x.to(device)

            # predicted_img, mask = model(img)
            loss = model(src_x,dst_x,src_dst_x)
            loss = loss / args.mask_ratio
            loss.backward()
            if step_count % steps_per_update == 0:
                optim.step()
                optim.zero_grad()
            losses.append(loss.item())
        lr_scheduler.step()
        avg_loss = sum(losses) / len(losses)
        #writer.add_scalar('mae_loss', avg_loss, global_step=e)
        print(f'In epoch {e}, average traning loss is {avg_loss}.')

        ''' save model '''
        torch.save(model, args.model_path)