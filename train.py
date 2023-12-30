import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from classifier.models import Classifier
from classifier.provider_cap3d import Cap3DDataset
from accelerate import Accelerator
from safetensors.torch import load_file

import kiui

def main(num_epochs=100, resume=None, workspace='workspace'):

    os.makedirs(workspace, exist_ok=True)
    accelerator = Accelerator(mixed_precision='fp16')

    # model
    model = Classifier()

    # resume
    if resume is not None:
        if resume.endswith('safetensors'):
            ckpt = load_file(resume, device='cpu')
        else:
            ckpt = torch.load(resume, map_location='cpu')
        model.load_state_dict(ckpt, strict=False)
    
    # data
    train_dataset = Cap3DDataset(training=True)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=512,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
    )

    test_dataset = Cap3DDataset(training=False)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )

    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=0.05, betas=(0.9, 0.95))

    # scheduler (per-iteration)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=3000, eta_min=1e-6)
    total_steps = num_epochs * len(train_dataloader)
    pct_start = 3000 / total_steps
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=4e-4, total_steps=total_steps, pct_start=pct_start)

    # accelerate
    model, optimizer, train_dataloader, test_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, test_dataloader, scheduler
    )

    # loop
    for epoch in range(num_epochs):
        # train
        model.train()
        total_loss = 0
        for i, data in enumerate(train_dataloader):
            with accelerator.accumulate(model):

                optimizer.zero_grad()

                step_ratio = (epoch + i / len(train_dataloader)) / num_epochs

                images = data['images']
                labels = data['labels']
                preds = model(images)

                loss = F.binary_cross_entropy_with_logits(preds, labels)
                accelerator.backward(loss)

                optimizer.step()
                scheduler.step()

                total_loss += loss.detach()

            if accelerator.is_main_process:
                if i % 10 == 0:
                    mem_free, mem_total = torch.cuda.mem_get_info()    
                    print(f"[INFO] {i}/{len(train_dataloader)} mem: {(mem_total-mem_free)/1024**3:.2f}/{mem_total/1024**3:.2f}G lr: {scheduler.get_last_lr()[0]:.7f} step_ratio: {step_ratio:.4f} loss: {loss.item():.6f}")
                
        total_loss = accelerator.gather_for_metrics(total_loss).mean()

        if accelerator.is_main_process:
            total_loss /= len(train_dataloader)
            accelerator.print(f"[train] epoch: {epoch} loss: {total_loss.item():.6f}")
        
        # checkpoint
        accelerator.wait_for_everyone()
        accelerator.save_model(model, workspace)

        # eval
        with torch.no_grad():
            model.eval()
            total_acc = 0
            for i, data in enumerate(test_dataloader):

                images = data['images']
                labels = data['labels']

                preds = model(images)
                preds = torch.sigmoid(preds) > 0.5
                
                accuracy = (preds == labels).float().mean()
                total_acc += accuracy.detach()
            
            total_acc = accelerator.gather_for_metrics(total_acc).mean()
            if accelerator.is_main_process:
                total_acc /= len(test_dataloader)
                accelerator.print(f"[train] epoch: {epoch} acc: {100 * total_acc.item():.2f}%")



if __name__ == "__main__":
    main()
