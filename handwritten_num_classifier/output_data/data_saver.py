
import shutil
from shutil import copytree, ignore_patterns

from pathlib import Path
import torch
import datetime
import csv

this_file_dir = Path(__file__).parent.resolve()
prj_root = this_file_dir / '../..'


class DataSaver(object):
    
    def __init__(self, config):
        super().__init__()    
        
        if config:
            if 'outputfile' in config:
                self.outputfile = config['outputfile']
        else:
            self.outputfile = prj_root / '.myoutput' / datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
        self.outputfile.mkdir(parents=True, exist_ok=True)

        self.metrics_master_dict = {}
        self.output_metrics_dir = self.outputfile / 'metrics'
        self.output_metrics_dir.mkdir(parents=True, exist_ok=True)
        self.output_metrics = self.output_metrics_dir / 'metrics.pkl'
        
        self.output_model_dir = self.outputfile / 'model'
        self.output_model_dir.mkdir(parents=True, exist_ok=True)
        
        self.save_code()
        

    def __del__(self):
        pass
    
    def save_code(self):
        src_root = prj_root / 'handwritten_num_classifier'
        self.output_code_dir = self.outputfile / 'code'
        copytree(str(src_root), 
                 str(self.output_code_dir), 
                 ignore=ignore_patterns('*.pyc', 
                                        '*__pycache__*'))

    def save_config(self, config_yml):
        filename = Path(config_yml).name
        (self.outputfile / 'config').mkdir(parents=True, exist_ok=True)
        self.output_config = self.outputfile / 'config' / filename
        shutil.copy(config_yml, str(self.output_config))
        
    def save_metrics(self, metrics_dict, epoch):
        self.metrics_master_dict.update({epoch: metrics_dict})
        torch.save(self.metrics_master_dict,
                   str(self.output_metrics))
        
        # 標準出力へoutput
        print('epoch {}, '.format(epoch, end=""))
        for metric_name, metric_val in metrics_dict.items():
            print('{}: {}, '.format(metric_name, metric_val), end="")
        print()  # 改行

    def save_model(self, model, epoch):
        modelname = '{}epoch.pth'.format(epoch)
        model_path = self.output_model_dir / modelname
        
        torch.save(model.state_dict(),
                   str(model_path))
        
    def save_result_detail(self, list_, epoch):
        self.output_result_detail = self.outputfile / 'detail'
        self.output_result_detail.mkdir(parents=True, exist_ok=True)
        self.output_result_csv = self.output_result_detail / f'{epoch}epoch_result_detail.csv'
        
        with self.output_result_csv.open('w') as f:
            writer = csv.writer(f)
            writer.writerow(['dataset_name', 'filename', 'gt', 'est'])
            for record in list_:
                writer.writerow(record)
        

    
"""

log

checkpoints

config

code

"""
