import numpy as np
import random

num_active = 12000
num_active_feat = 25
num_lost = 8000
num_lost_feat = 26
num_edge_feat = 1

train_num_pairs = 5000

val_num_pairs = 2000

unlabeled_num_pairs = 20000

active_players = []
for i in range(num_active):
    active_players.append("ActivePlayer_" + str(i))    

active_player_features = np.random.rand(num_active, num_active_feat)

lost_players = []
for j in range(num_lost):
    lost_players.append("LostPlayer_" + str(j))

lost_player_features = np.random.rand(num_lost, num_lost_feat)

player_pairs = set()

train_lines_to_write = []


train_active_ids = set()
val_active_ids = set()

for k in range(train_num_pairs):
    k_a = np.random.randint(num_active)
    k_l = np.random.randint(num_lost)

    active_id = active_players[k_a]
    lost_id = lost_players[k_l]

    pair_id = active_id + "," + lost_id
    if pair_id in player_pairs:
        continue

    train_active_ids.add(active_id)    
    player_pairs.add(pair_id)

    active_feat = active_player_features[k_a]
    lost_feat = lost_player_features[k_l]

    edge_feat = np.random.rand(1, num_edge_feat)

    id_cols = [active_id, lost_id]

    check_cols = ["0","0","0","0"]

    labels = []
    if random.random() < 0.05:
        label = "1"
    else:
        label = "0"

    labels.append(label)    


    a_feat_cols =[str(x) for x in active_feat]   

    ab_feat_cols = [str(x) for x in edge_feat[0]]

    b_feat_cols = [str(x) for x in lost_feat]

    line_cols = id_cols + check_cols + labels + a_feat_cols + ab_feat_cols + b_feat_cols
    
    line_to_write = "\t".join(line_cols)

    train_lines_to_write.append(line_to_write + "\n")

with open("data/train", "w") as f:
    f.writelines(train_lines_to_write)

val_lines_to_write = []

for k in range(val_num_pairs):
    k_a = np.random.randint(num_active)
    k_l = np.random.randint(num_lost)

    active_id = active_players[k_a]
    lost_id = lost_players[k_l]

    pair_id = active_id + "," + lost_id
    if pair_id in player_pairs:
        continue

    if active_id in train_active_ids:
        continue  

    val_active_ids.add(active_id)    
    player_pairs.add(pair_id)

    active_feat = active_player_features[k_a]
    lost_feat = lost_player_features[k_l]

    edge_feat = np.random.rand(1, num_edge_feat)

    id_cols = [active_id, lost_id]

    check_cols = ["0","0","0","0"]

    labels = []
    if random.random() < 0.05:
        label = "1"
    else:
        label = "0"

    labels.append(label)    


    a_feat_cols =[str(x) for x in active_feat]   

    ab_feat_cols = [str(x) for x in edge_feat[0]]

    b_feat_cols = [str(x) for x in lost_feat]

    line_cols = id_cols + check_cols + labels + a_feat_cols + ab_feat_cols + b_feat_cols
    
    line_to_write = "\t".join(line_cols)

    val_lines_to_write.append(line_to_write + "\n")

with open("data/val", "w") as f:
    f.writelines(val_lines_to_write)


unlabeled_lines_to_write = []

for k in range(unlabeled_num_pairs):
    k_a = np.random.randint(num_active)
    k_l = np.random.randint(num_lost)

    active_id = active_players[k_a]
    lost_id = lost_players[k_l]

    pair_id = active_id + "," + lost_id
    if pair_id in player_pairs:
        continue

    if active_id in train_active_ids or active_id in val_active_ids:
        continue  

    val_active_ids.add(active_id)    
    player_pairs.add(pair_id)

    active_feat = active_player_features[k_a]
    lost_feat = lost_player_features[k_l]

    edge_feat = np.random.rand(1, num_edge_feat)

    id_cols = [active_id, lost_id]

    check_cols = ["0","0","0","0"]

    labels = []
    label = "0"
    labels.append(label)    


    a_feat_cols =[str(x) for x in active_feat]   

    ab_feat_cols = [str(x) for x in edge_feat[0]]

    b_feat_cols = [str(x) for x in lost_feat]

    line_cols = id_cols + check_cols + labels + a_feat_cols + ab_feat_cols + b_feat_cols
    
    line_to_write = "\t".join(line_cols)

    unlabeled_lines_to_write.append(line_to_write + "\n")

with open("data/unlabeled", "w") as f:
    f.writelines(unlabeled_lines_to_write)