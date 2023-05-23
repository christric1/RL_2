import argparse
from pathlib import Path
from tqdm import tqdm
import psutil
import os
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

from utils.general import increment_path

from ddpg_agent import DDPGAgent
from dataset import OD_Dataset
from detect import yolo
from reinforcement import *


def record_training(save_dir, epochs, updateCnt):
    doc = "record.txt"
    with open(os.path.join(save_dir, doc), 'w') as f:
        f.write(str(epochs) + " " + str(updateCnt))


def print_memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    print(f"Resident memory (RSS): {mem_info.rss / 1024**2:.2f} MB")
    print(f"Virtual memory (VMS): {mem_info.vms / 1024**2:.2f} MB")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-path', type=str, default='coco')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--steps', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=32, help='total batch size for all GPUs')
    parser.add_argument('--buffer-size', type=int, default=2000, help='buffer size')
    parser.add_argument('--project', default='runs/train', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--resume', type=str, help='resume?')
    parser.add_argument('--plot', action='store_true', help='plot?')
    parser.add_argument('--split_dataset', default='1000', type=int, help='split dataset')
    opt = parser.parse_args()

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device: ", device)

    # Resume
    if opt.resume:
        save_dir = opt.resume
        writer = SummaryWriter(log_dir=save_dir + "/metrics")

        action_dim = 4
        agent = DDPGAgent(action_dim, save_dir, buffer_size=opt.buffer_size, batch_size=opt.batch_size)
        agent.load(save_dir)
        
        with open(os.path.join(save_dir, "record.txt")) as f:
            data = f.readline().split()
            init_epoch, update_cnt = int(data[0]), int(data[1])

    else:
        # Result directary
        save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=False)  # increment run
        writer = SummaryWriter(save_dir + "/metrics")

        # Create dqn model & YOLOv7
        action_dim = 4
        agent = DDPGAgent(action_dim, save_dir, buffer_size=opt.buffer_size, batch_size=opt.batch_size)
        init_epoch, update_cnt = 0, 0

    # yolov7 model
    yolo_model = yolo()

    # Trainloader & Testloader
    trainDataset = OD_Dataset(opt.dataset_path, mode='train')
    trainDataloader = DataLoader(trainDataset, batch_size=1, shuffle=True)

    #---------------------------------------#
    #   Start training
    #---------------------------------------#
    for epoch in range(init_epoch, opt.epochs):
        '''
            Random split dataset
        '''
        if opt.split_dataset:
            subset_indices = random.sample(range(len(trainDataset)), opt.split_dataset)
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
                Avg_iou_origin, precision_origin, recall_origin = yolo_model.detectImg(img, target)
                Origin_score = get_score(Avg_iou_origin, precision_origin, recall_origin)

                # Distortion Image
                distortion_img = distortion_image(img)
                Avg_iou_distortion, precision_distortion, recall_distortion = yolo_model.detectImg(distortion_img, target)
                Distortion_score = get_score(Avg_iou_distortion, precision_distortion, recall_distortion)

                # With RL Score
                action = agent.act(distortion_img.unsqueeze(dim=0))
                trans_action = transform_action(action[0], 0.7, 1.3)
                adjust_img = modify_image(img, *trans_action)
                Avg_iou_RL, precision_score_RL, recall_distortion_RL = yolo_model.detectImg(adjust_img, target)
                RL_score = get_score(Avg_iou_RL, precision_score_RL, recall_distortion_RL)

                reward = get_reward(RL_score, Origin_score, Distortion_score, 0.01)
                
                # Push experient to memory
                state = distortion_img
                next_state = adjust_img
                critic_loss, actor_loss = agent.step(state, action, reward, next_state)
                
                # Move to the next state
                state = next_state

                # Print & Record
                pbar.set_description((f"Epoch [{epoch+1}/{opt.epochs}]"))
                pbar.set_postfix({"Critic_loss": "{:.4f}".format(critic_loss),
                                  "Actor_loss": "{:.4f}".format(actor_loss)})
                
                # Writer
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
        agent.save()    # Save model
        record_training(save_dir, epoch+1, update_cnt)    # Save training record

    # end epoch ---------------------------------------------------------

    # End training ---------------------------------------------------------
    print("End Training\n")