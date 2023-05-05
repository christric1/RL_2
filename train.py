import argparse
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

from utils.general import increment_path

from ddpg_agent import DDPGAgent
from dataset import OD_Dataset
from detect import yolo
from reinforcement import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='models/yolov7_backbone_weights.pth', help='initial weights path')
    parser.add_argument('--hyp', type=str, default='data/hyp.yaml', help='hyperparameters path')
    parser.add_argument('--dataset-path', type=str, default='Pascal_2012')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--steps', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=32, help='total batch size for all GPUs')
    parser.add_argument('--buffer-size', type=int, default=2e3, help='buffer size')
    parser.add_argument('--project', default='runs/train', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--plot', action='store_true', help='plot?')
    parser.add_argument('--no-writer', dest='writer', action='store_false', help='writer?')
    parser.set_defaults(writer=True)
    opt = parser.parse_args()

    # Result directary
    save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=False)  # increment run
    writer = SummaryWriter(save_dir + "/metrics")

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device: ", device)

    # Create dqn model & YOLOv7
    obs_dim, action_dim = 8*8*1024 + 6*4, 3
    agent = DDPGAgent(action_dim, save_dir, memory_size=opt.buffer_size, batch_size=opt.batch_size)
    yolo_model = yolo()

    # Trainloader & Testloader
    trainDataset = OD_Dataset(opt.dataset_path, mode='train')
    # valDataset = OD_Dataset(opt.dataset_path, mode='valid')

    #---------------------------------------#
    #   Start training
    #---------------------------------------#
    update_cnt = 0
    for epoch in range(opt.epochs):
        '''
            Random split dataset
        '''
        subset_indices = random.sample(range(len(trainDataset)), 1000)
        partial_dataset = Subset(trainDataset, subset_indices)
        trainDataloader = DataLoader(partial_dataset, batch_size=1, shuffle=True)

        pbar = tqdm(enumerate(trainDataloader), total=len(trainDataloader))
        for i, data in pbar:
            '''
                img     : (3, height, width)
                target  : [label, xmin, ymin, xmax, ymax]
            '''
            img, target, img_path = data
            img = img.squeeze(dim=0)
            target = target.squeeze(dim=0)
            labels, boxs = target[:, 0], target[:, 1:]

            for step in range(opt.steps):
                '''
                    action : [contrast, saturation, brightness]
                '''
                # Origin Image
                Avg_iou_origin, F1_score_origin = yolo_model.detectImg(img, target)
                Origin_score = get_score(Avg_iou_origin, F1_score_origin, GAMMA)

                # Distortion Image
                distortion_img = distortion_image(img)
                Avg_iou_distortion, F1_score_distortion = yolo_model.detectImg(distortion_img, target)
                Distortion_score = get_score(Avg_iou_distortion, F1_score_distortion, GAMMA)

                # With RL Score
                action = agent.act(distortion_img.unsqueeze(dim=0))
                trans_action = transform_action(action[0], [0.5, 1.5])
                adjust_img = modify_image(img, *trans_action)
                Avg_iou_RL, F1_score_RL = yolo_model.detectImg(adjust_img, target)
                RL_score = get_score(Avg_iou_RL, F1_score_RL, GAMMA)

                reward = get_reward(RL_score, Origin_score, Distortion_score)
                
                # Push experient to memory
                state = img
                next_state = adjust_img
                critic_loss, actor_loss = agent.step(state, action, reward, next_state)
                
                # Move to the next state
                state = next_state

                # Print & Record
                pbar.set_description((f"Epoch [{epoch+1}/{opt.epochs}]"))
                pbar.set_postfix({"Critic_loss": "{:.4f}".format(critic_loss),
                                  "Actor_loss": "{:.4f}".format(actor_loss)})
                
                if opt.writer:
                    writer.add_scalar('Critic_loss/train', critic_loss, update_cnt)
                    writer.add_scalar('Actor_loss/train', actor_loss, update_cnt)
                    writer.add_scalar('Reward/train', reward, update_cnt)
                    update_cnt += 1 

                if opt.plot:
                    fig, axs = plt.subplots(1, 3, figsize=(10, 5))
                    axs[0].imshow(TF.to_pil_image(img))
                    axs[0].set_title('origin image')   
                    axs[1].imshow(TF.to_pil_image(distortion_img))
                    axs[1].set_title('distortion image')   
                    axs[2].imshow(TF.to_pil_image(adjust_img))
                    axs[2].set_title('adjust image')      
                    plt.savefig('image.jpg')
        # end batch -------------------------------------------------------------
    # end epoch ---------------------------------------------------------
    del trainDataloader, partial_dataset

    # Validation
    # agent.eval()

    # End training ---------------------------------------------------------
    print("End Training\n")
    
    # Save model
    agent.save()