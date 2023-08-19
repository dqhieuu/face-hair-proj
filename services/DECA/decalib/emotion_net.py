import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


# FLAME2020: shape(100), exp(50), pose(15 = (global_pose(3), neck_pose(3), jaw_pose(3), eye_pose(6)))
# DECA 6 pose: (jaw_pose(3), global_pose(3)) (global pose in most cases isn't used)
# values ranging from -1 to 1

# network: 100 shape + 6 my emotion -> 50 expression + 6 pose
# loss: L2_sum(50 exp) + L2_sum(first 3 pose)*20 + L2_sum(other 3 pose)

class MyNet(nn.Module):
  def __init__(self):
    super().__init__()
    self.mlp = nn.Sequential(
        nn.Linear(106, 128),
        nn.ReLU(),
        nn.Linear(128, 56),
    )

    self.net = nn.Sequential(
        self.mlp
    )

  # shape_params: B * 100, my_emotion_params: B * 6
  def forward(self, shape_params, my_emotion_params):
    combined_input_params = torch.cat((shape_params, my_emotion_params), dim=1)
    output = self.net(combined_input_params)
    #first 50 is exp, last 6 is pose
    exp_params, pose_params = output[:, :50], output[:, 50:]
    return exp_params, pose_params

  @staticmethod
  def my_loss(pred, truth):
    pred_exp, pred_pose_jaw, pred_pose_global = pred[:, 0:50], pred[:, 50:53], pred[:, 53:]
    truth_exp, truth_pose_jaw, truth_pose_global = truth[:, 0:50], truth[:, 50:53], truth[:, 53:]
    mae = nn.L1Loss()
    loss_exp = mae(pred_exp, truth_exp)
    loss_pose_global = mae(pred_pose_global, truth_pose_global)
    loss_pose_jaw = mae(pred_pose_jaw, truth_pose_jaw) * 20 # jaw params are important

    total_loss = loss_exp + loss_pose_global + loss_pose_jaw
    return total_loss

  @staticmethod
  def result_insight(pred, truth):
    first_pred, first_truth = pred[0], truth[0]
    pred_exp, pred_pose_jaw, pred_pose_global = first_pred[0:50], first_pred[50:53], first_pred[53:]
    truth_exp, truth_pose_jaw, truth_pose_global = first_truth[0:50], first_truth[50:53], first_truth[53:]
    pred_min = torch.min(first_pred)
    pred_max = torch.max(first_truth)
    pred_mean = torch.mean(first_pred)
    pred_std = torch.std(first_pred)
    truth_min = torch.min(first_truth)
    truth_max = torch.max(first_truth)
    truth_mean = torch.mean(first_truth)
    truth_std = torch.std(first_truth)
    all_diff = torch.abs(first_pred - first_truth)
    exp_diff = torch.abs(pred_exp - truth_exp)
    jaw_pose_diff = torch.abs(pred_pose_jaw - truth_pose_jaw)
    global_pose_diff = torch.abs(pred_pose_global - truth_pose_global)
    min_diff = torch.min(all_diff)
    max_diff = torch.max(all_diff)
    mean_diff = torch.mean(all_diff)
    exp_min_diff = torch.min(exp_diff)
    exp_max_diff = torch.max(exp_diff)
    exp_mean_diff = torch.mean(exp_diff)
    jaw_pose_min_diff = torch.min(jaw_pose_diff)
    jaw_pose_max_diff = torch.max(jaw_pose_diff)
    jaw_pose_mean_diff = torch.mean(jaw_pose_diff)
    global_pose_min_diff = torch.min(global_pose_diff)
    global_pose_max_diff = torch.max(global_pose_diff)
    global_pose_mean_diff = torch.mean(global_pose_diff)

    return {
        'pred_min': pred_min,
        'pred_max': pred_max,
        'pred_mean': pred_mean,
        'pred_std': pred_std,
        'truth_min': truth_min,
        'truth_max': truth_max,
        'truth_mean': truth_mean,
        'truth_std': truth_std,
        'min_diff': min_diff,
        'max_diff': max_diff,
        'mean_diff': mean_diff,
        'exp_min_diff': exp_min_diff,
        'exp_max_diff': exp_max_diff,
        'exp_mean_diff': exp_mean_diff,
        'jaw_pose_min_diff': jaw_pose_min_diff,
        'jaw_pose_max_diff': jaw_pose_max_diff,
        'jaw_pose_mean_diff': jaw_pose_mean_diff,
        'global_pose_min_diff': global_pose_min_diff,
        'global_pose_max_diff': global_pose_max_diff,
        'global_pose_mean_diff': global_pose_mean_diff,
    }

def get_emotion_model():
  torch_model = MyNet().to(device)
  checkpoint = torch.load("decalib/emotion_net-epoch=499-step=6000.ckpt", map_location=device)
  model_weights = checkpoint["state_dict"]
  for key in list(model_weights):
      model_weights[key.replace("model.", "")] = model_weights.pop(key)
  torch_model.load_state_dict(model_weights)
  torch_model.eval()

  return torch_model

# test for an example data
if __name__ == '__main__':
    model = get_emotion_model()

    with open('../emotion_train_data/V000131_S002_L2_E01_C7.json', 'r') as f:
      data = json.load(f)
      res = model(torch.tensor([data['shape']]), torch.tensor([data['my_emotion']]))
      print(res)
