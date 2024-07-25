import argparse
import csv
import json
import logging
import os
import random
import tempfile
import torch.distributed as dist
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
import os.path as path
import time
import sys
import subprocess

import losses
import model
from evaluation.evaler import Evaler
from model import att_basic_model
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms

import Datasets
from cococaption.pycocoevalcap.eval import COCOEvalCap
from cococaption.pycocotools.coco import COCO
from lib.config import cfg, cfg_from_file
from optimizer.optimizer import Optimizer
from samplers import distributed


try:
    from nltk.translate.bleu_score import corpus_bleu
except:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", "nltk"])

from nltk.translate.bleu_score import corpus_bleu
from Datasets import dataset, data_loader
from models import *
from lib.utils import *

logging.basicConfig(level=logging.INFO)

logging.info(torch.cuda.is_available())

class Trainer(object):
    def __init__(self, args):
        super(Trainer, self).__init__()
        self.args = args
        if cfg.SEED > 0:
            random.seed(cfg.SEED)
            torch.manual_seed(cfg.SEED)
            torch.cuda.manual_seed_all(cfg.SEED)

        self.num_gpus = 1
        self.setup_logging()
        self.distributed = self.num_gpus > 1
        if self.distributed:
            torch.cuda.set_device(args.local_rank)
            torch.distributed.init_process_group(
                backend="nccl", init_method="env://"
            )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.setup_dataset()
        self.setup_network()
        self.vocab_size = cfg.MODEL.VOCAB_SIZE + 1
        self.logit=nn.Linear(cfg.MODEL.RNN_SIZE, self.vocab_size).to(self.device)
        self.xe_criterion = losses.create(cfg.LOSSES.XE_TYPE).to(self.device)
        self.val_evaler = Evaler(self.args,
            eval_ids=cfg.DATA_LOADER.VAL_ID,
            gv_feat=cfg.DATA_LOADER.VAL_GV_FEAT,
            att_feats=cfg.DATA_LOADER.VAL_ATT_FEATS,
            eval_annfile=cfg.INFERENCE.VAL_ANNFILE
        )


    def setup_dataset(self):
        self.set = dataset.Dataset(
            image_ids_path = cfg.DATA_LOADER.TRAIN_ID,
            input_seq = cfg.DATA_LOADER.INPUT_SEQ_PATH,
            target_seq = cfg.DATA_LOADER.TARGET_SEQ_PATH,
            gv_feat_path = cfg.DATA_LOADER.TRAIN_GV_FEAT,
            att_feats_folder = cfg.DATA_LOADER.TRAIN_ATT_FEATS,
            seq_per_img = cfg.DATA_LOADER.SEQ_PER_IMG,
            max_feat_num = cfg.DATA_LOADER.MAX_FEAT,
            dis=None
        )
    def setup_network(self):
        generator=model.create(cfg.MODEL.TYPE)
        if self.distributed:
            # this should be removed if we update BatchNorm stats
            self.generator = torch.nn.parallel.DistributedDataParallel(
                generator.to(self.device),
                device_ids = [self.args.local_rank],
                output_device = self.args.local_rank,
                broadcast_buffers = False
            )
        else:
            self.generator = generator.to(self.device)
        checkpoint_path = args.storage + '/ckpts/' + args.dataset + '/gen/' + args.checkpoint_filename
        if path.isfile(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            self.generator.load_state_dict(checkpoint['gen_state_dict'])
            self.generator.to(self.device)
        self.optim = Optimizer(self.generator)

    def setup_loader_train(self,  epoch):
        self.training_loader = data_loader.load_train(
            self.distributed, epoch, self.set)

    def make_kwargs(self,indices, input_seq, target_seq, gv_feat, att_feats, att_mask):
        seq_mask = (input_seq > 0).type(torch.LongTensor)
        seq_mask[:, 0] += 1
        seq_mask_sum = seq_mask.sum(-1)
        max_len = int(seq_mask_sum.max())
        input_seq = input_seq[:, 0:max_len].contiguous()
        target_seq = target_seq[:, 0:max_len].contiguous()

        kwargs = {
            cfg.PARAM.INDICES: indices,
            cfg.PARAM.INPUT_SENT: input_seq,
            cfg.PARAM.TARGET_SENT: target_seq,
            cfg.PARAM.GLOBAL_FEAT: gv_feat,
            cfg.PARAM.ATT_FEATS: att_feats,
            cfg.PARAM.ATT_FEATS_MASK: att_mask
        }
        return kwargs, max_len


    def scheduled_sampling(self, epoch):
        if epoch > cfg.TRAIN.SCHEDULED_SAMPLING.START:
            frac = (epoch - cfg.TRAIN.SCHEDULED_SAMPLING.START) // cfg.TRAIN.SCHEDULED_SAMPLING.INC_EVERY
            ss_prob = min(cfg.TRAIN.SCHEDULED_SAMPLING.INC_PROB * frac, cfg.TRAIN.SCHEDULED_SAMPLING.MAX_PROB)
            self.generator.ss_prob = ss_prob
            
    def setup_logging(self):
        self.logger = logging.getLogger(cfg.LOGGER_NAME)
        self.logger.setLevel(logging.INFO)
        
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter("[%(levelname)s: %(asctime)s] %(message)s")
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)        
        fh = logging.FileHandler(os.path.join('./stats/LGK/gen/mle/', cfg.LOGGER_NAME + '.txt'))
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)


    def gen_mle_train(self,epoch):
        # 将生成器模块设置为训练模式
        self.generator.train()
        losses = AverageMeter()
        top5 = AverageMeter()
        top1 = AverageMeter()
        self.setup_loader_train(epoch)
        for batch_id, (indices, input_seq, target_seq, gv_feat, att_feats) in enumerate(self.training_loader):
            start_time = time.time()  # 计时

            atts_num = [x.shape[0] for x in att_feats]
            max_att_num = np.max(atts_num)
            mask_arr = []
            for i, num in enumerate(atts_num):
                tmp_mask = np.zeros((1, max_att_num), dtype=np.float32)
                tmp_mask[:, 0:num] = 1
                mask_arr.append(torch.from_numpy(tmp_mask))
            att_mask = torch.cat(mask_arr, 0)

            input_seq = input_seq.to(self.device)
            target_seq = target_seq.to(self.device)
            gv_feat = gv_feat.to(self.device)
            att_feats = att_feats.to(self.device)
            att_mask = att_mask.to(self.device)

            kwargs, max_len = self.make_kwargs(indices, input_seq, target_seq, gv_feat, att_feats, att_mask)
            preds = self.generator(**kwargs).to(self.device)
            #print(preds)
            #print(preds.size(-1))
            #print(kwargs[cfg.PARAM.TARGET_SENT])

            loss, loss_info = self.xe_criterion(preds, kwargs[cfg.PARAM.TARGET_SENT])
            Preds = preds.view(-1, preds.shape[-1])
            Target_seq = kwargs[cfg.PARAM.TARGET_SENT].view(-1)
            #print(Preds.size())
            #print(Target_seq.size())
            loss.backward()
            utils.clip_gradient(self.optim.optimizer, self.generator, cfg.SOLVER.GRAD_CLIP_TYPE, cfg.SOLVER.GRAD_CLIP)
            self.optim.step()
            self.optim.zero_grad()
            self.optim.scheduler_step('Iter')

            top1_acc = categorical_accuracy(Preds, Target_seq, 1)
            #print(top1_acc)
            top5_acc = categorical_accuracy(Preds, Target_seq, 5)
            #print(top5_acc)
            top1.update(top1_acc)
            top5.update(top5_acc)

            batch_time = time.time() - start_time
            losses.update(loss.item())
            self.iteration += 1
            #print(losses)
            # 打印日志

            if batch_id % args.print_freq == 0:
                logging.info(
                             'Epoch: [{}]\t'
                             'Batch: [{}]\t'
                             'Time per batch: [{:.3f}]\t'
                             'Loss [{:.5}]\t'
                             'Top 1 accuracy [{:.4f}]\t'
                             'Top 5 accuracy [{:.4f}]\t'
                                 .format(epoch, batch_id, batch_time, losses.avg,top1.avg,top5.avg))

                if args.save_stats:
                    with open(args.storage + '/stats/' + args.dataset + '/gen/TRAIN_MLE_GEN.csv', 'a+') as file:
                        writer = csv.writer(file)
                        writer.writerow([epoch, batch_id, batch_time, losses.avg,top1.avg,top5.avg])

    def val(self, epoch):
        val_res = self.val_evaler(self.generator, 'val_' + str(epoch))
        #logging.info('######## Epoch (VAL)' + str(epoch) + ' ########\t')
        #logging.info(str(val_res))
        self.logger.info('######## Epoch(VAL) ' + str(epoch) + ' ########\t')
        self.logger.info(str(val_res))
        val = 0
        for score_type, weight in zip(cfg.SCORER.TYPES, cfg.SCORER.WEIGHTS):
            val -= val_res[score_type] * weight
        return val

    def train(self):
        self.optim.zero_grad()
        self.iteration = 0

        for e in range(args.epochs):
            self.gen_mle_train(e)
            val=self.val(e)
            if args.save_model:
                torch.save({'gen_state_dict': self.generator.state_dict()}, args.storage + '/ckpts/' + args.dataset + '/gen/{}_{}.pth'.format('MLE_GEN', e))
            self.optim.scheduler_step('Epoch', val)
            self.scheduled_sampling(e)
            


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Maximum Likelihood Estimation Training')
    parser.add_argument('--print-freq', type=int, default=50)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--storage', type=str, default='.')
    #parser.add_argument('--dataset', type=str, default='CXR')
    parser.add_argument('--dataset', type=str, default='LGK')
    parser.add_argument('--checkpoint-filename', type=str, default='')
    parser.add_argument('--save-model', type=bool, default=True)
    parser.add_argument('--save-stats', type=bool, default=True)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--folder', dest='folder', type=str, default='./')
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--result-folder", type=str, default='./stats/LGK/gen/mle/result')
    args=parser.parse_args()
    cfg_from_file(os.path.join(args.folder, 'configmle.yml'))
    #cfg_from_file(os.path.join(args.folder, 'configtrans.yml'))
    trainer = Trainer(args)
    trainer.train()

