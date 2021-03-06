# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import datetime
import logging
import time

import torch
import torch.distributed as dist

from maskrcnn_benchmark.utils.comm import get_world_size
from maskrcnn_benchmark.utils.metric_logger import MetricLogger

import ipdb
import time

import numpy as np
import cv2
import matplotlib.pyplot as plt
from torchvision.transforms import transforms
from torchvision.transforms import ToPILImage

def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k, v in loss_dict.items():
            loss_names.append(k)
            all_losses.append(v)
        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses


def do_train(
    model,
    data_loader,
    optimizer,
    scheduler,
    checkpointer,
    device,
    checkpoint_period,
    arguments,
):
    logger = logging.getLogger("maskrcnn_benchmark.trainer")
    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")
    max_iter = len(data_loader)
    print(max_iter)
    start_iter = arguments["iteration"]
    print(start_iter)
    model.train()
    start_training_time = time.time()
    end = time.time()

    for iteration, (images, targets, _) in enumerate(data_loader, start_iter):
        data_time = time.time() - end
        iteration = iteration + 1
        arguments["iteration"] = iteration

        #ipdb.set_trace()
        #unloader=transforms.ToPILImage()
        #showimg = unloader(images.tensors[0])
        #showimg = np.array(showimg)
        #bboxs=targets[0].bbox
        #bboxs = bboxs.numpy()
        #for bbox in bboxs:
        #    cv2.rectangle(showimg, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 4)
        #plt.imshow(showimg)
        #for mask in targets[0].extra_fields['masks'].polygons:
        #    poly = mask.polygons
        #    poly = poly[0]
        #    #poly = poly.numpy()
        #    n = len(poly)
        #    x = []
        #    y = []
        #    for i in range(int(n/2)):
        #        x.append(int(poly[i*2]))
        #        y.append(int(poly[i*2+1]))
        #    plt.plot(x,y,color="red",linewidth=2.0)
        #    #plt.scatter(x, y, color = 'red')
        #ipdb.set_trace()
        #plt.imshow(showimg)

        #plt.show()


        scheduler.step()

        #time.sleep( 1 )
        images = images.to(device)
        targets = [target.to(device) for target in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_loss_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        meters.update(loss=losses_reduced, **loss_dict_reduced)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)

        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if iteration % 20 == 0 or iteration == max_iter:
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters),
                    lr=optimizer.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )
        if iteration % checkpoint_period == 0:
            checkpointer.save("model_{:07d}".format(iteration), **arguments)

    checkpointer.save("model_{:07d}".format(iteration), **arguments)
    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )
