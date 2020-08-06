
import sys
import os

import yaml
import torch
import torchvision
import torch.nn as nn
from tqdm import tqdm

import sys
from pathlib import Path
this_file_dir = Path(__file__).parent.resolve()

sys.path += [
    str(this_file_dir),
]

from models.model_factory import get_model, load_weight
from datasets.dataset_factory import get_dataloader_through_dataset
from losses.loss_factory import get_loss
from optimizers.optimizer_factory import get_optimizer
from output_data.data_saver import DataSaver


def train(config_yml, 
          working_root=str(this_file_dir / '..')):

    with open(config_yml, 'r') as f:
        config = yaml.safe_load(f)

    # ====
    # データ用意
    # ====
    train_loader = get_dataloader_through_dataset(
        config['data']['train'],
        working_root
    )
    test_loader  = get_dataloader_through_dataset(
        config['data']['eval'],
        working_root,
    )

    # ネットワーク用意
    net = get_model(config['model'])
    if config['model'].get('model_state_dict'):
        model_state_dict_path = Path(working_root) / config['model']['model_state_dict']
        load_weight(net, str(model_state_dict_path))
    
    # optimizer定義
    optimizer = get_optimizer(net, config['optimizer'])

    # 損失関数定義
    criterion = get_loss(config['loss'])

    # データを保存する機能を持つオブジェクト
    datasaver = DataSaver(config['output_data'])
    datasaver.save_config(config_yml)

    # ======
    # メインループ
    # ======
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if 'cuda' in config:
        device = 'cuda' if config['cuda'] == True else 'cpu'

    net.to(device)

    num_epochs = config['num_epochs']
    for epoch in range(num_epochs):
        print(epoch)
        metrics_dict = {}

        #train
        print('train phase')
        metrics = run_train(net, train_loader, criterion, optimizer, device)
        metrics_dict.update(metrics)

        # eval
        print('eval phase')
        metrics, result_detail = run_eval(net, test_loader, criterion, device)
        metrics_dict.update(metrics)

        # 評価指標の記録
        datasaver.save_metrics(metrics_dict, epoch)
        datasaver.save_model(net, epoch)
        datasaver.save_result_detail(result_detail, epoch)


def run_train(model, data_loader, criterion, optimizer, device, grad_acc=1):
    model.train()

    # zero the parameter gradients
    optimizer.zero_grad()

    total_loss = 0.
    train_acc = 0
    for i, (inputs, gt_labels, _, _) in tqdm(enumerate(data_loader), total=len(data_loader)):
        
        # 画像が合っているかを確認
        if os.environ.get('DEBUG', None) is not None:
            import cv2
            im = (inputs.numpy()*255)[0].transpose(1,2,0)
            cv2.imwrite('./sample.png', im)
        
        inputs = inputs.to(device)
        gt_labels = gt_labels.to(device)

        outputs = model(inputs)

        loss = criterion(outputs, gt_labels)
        loss.backward()

        # Gradient accumulation
        if (i % grad_acc) == 0:
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item()
        train_acc += (outputs.max(1)[1] == gt_labels).sum().item()

    total_loss /= len(data_loader)
    avg_train_acc = train_acc / len(data_loader.dataset)
    
    metrics = {'train_loss': total_loss,
               'train_acc': avg_train_acc}

    return metrics


def run_eval(model, data_loader, criterion, device):
    model.eval()
    val_acc = 0.
    
    result_detail = []

    with torch.no_grad():
        total_loss = 0.
        for inputs, gt_labels, filenames, dataset_name in tqdm(data_loader, total=len(data_loader)):
            inputs = inputs.to(device)
            gt_labels = gt_labels.to(device)

            outputs = model(inputs)
            est = outputs.max(1)[1]

            loss = criterion(outputs, gt_labels)

            total_loss += loss.item()
            correct = (est == gt_labels)
            val_acc += correct.sum().item()
    
            l = []
            for i in range(len(filenames)):
                gt_label = model.class_labels[gt_labels[i].data.item()]
                est_label = model.class_labels[est.data[i].item()]
                result_detail.append([dataset_name[i],
                                      filenames[i], 
                                      gt_label, 
                                      est_label])
            
        total_loss /= len(data_loader)  # iterした回数で割る
        val_acc /= len(data_loader.dataset)  # 画像の総枚数で割る
        metrics = {'val_loss': total_loss, 
                   'val_acc': val_acc}

    return metrics, result_detail


if __name__ == '__main__':
    argvs = sys.argv
    train(argvs[1], argvs[2])

