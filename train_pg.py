import argparse
import csv
import json
import logging
import os
import os.path as path
import random
import time

import model
from lib.config import cfg, cfg_from_file
import torch.optim as optim
from nltk.translate.bleu_score import corpus_bleu
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
import torch.distributed as dist
from Datasets import dataset, data_loader
import losses
from datasets import ImageCaptionDataset
from evaluation.evaler import Evaler
from models import *

from lib.utils import *
from scorer.scorer import Scorer
from optimizer.optimizer import Optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(torch.cuda.is_available())
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
])

logging.basicConfig(level=logging.INFO)

class Trainer(object):
    def __init__(self, args):
        super(Trainer, self).__init__()
        self.args = args
        if cfg.SEED > 0:
            random.seed(cfg.SEED)
            torch.manual_seed(cfg.SEED)
            torch.cuda.manual_seed_all(cfg.SEED)

        self.num_gpus = 1
        self.distributed = self.num_gpus > 1
        if self.distributed:
            torch.cuda.set_device(args.local_rank)
            torch.distributed.init_process_group(
                backend="nccl", init_method="env://"
            )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vocab_size = cfg.MODEL.VOCAB_SIZE + 1
        self.setup_dataset()
        self.setup_network()
        self.logit=nn.Linear(cfg.MODEL.RNN_SIZE, self.vocab_size).to(self.device)
        self.xe_criterion = losses.create(cfg.LOSSES.XE_TYPE).to(self.device)
        self.val_evaler = Evaler(self.args,
            eval_ids=cfg.DATA_LOADER.VAL_ID,
            gv_feat=cfg.DATA_LOADER.VAL_GV_FEAT,
            att_feats=cfg.DATA_LOADER.VAL_ATT_FEATS,
            eval_annfile=cfg.INFERENCE.VAL_ANNFILE
        )
        self.scorer = Scorer()

    def setup_dataset(self):
        self.set = dataset.Dataset(
            image_ids_path=cfg.DATA_LOADER.TRAIN_ID,
            input_seq=cfg.DATA_LOADER.INPUT_SEQ_PATH,
            target_seq=cfg.DATA_LOADER.TARGET_SEQ_PATH,
            gv_feat_path=cfg.DATA_LOADER.TRAIN_GV_FEAT,
            att_feats_folder=cfg.DATA_LOADER.TRAIN_ATT_FEATS,
            seq_per_img=cfg.DATA_LOADER.SEQ_PER_IMG,
            max_feat_num=cfg.DATA_LOADER.MAX_FEAT,
            dis=None
        )

    def setup_network(self):
        generator = model.create(cfg.MODEL.TYPE)
        if self.distributed:
            # this should be removed if we update BatchNorm stats
            self.generator = torch.nn.parallel.DistributedDataParallel(
                generator.to(self.device),
                device_ids=[self.args.local_rank],
                output_device=self.args.local_rank,
                broadcast_buffers=False
            )
        else:
            self.generator = generator.to(self.device)
        self.discriminator = GRUDiscriminator(embedding_dim=args.dis_embedding_dim,
                                              gru_units=args.dis_gru_units,
                                              vocab_size=self.vocab_size,
                                              encoder_dim=2048)
        self.discriminator.to(self.device)
        self.evaluator = Evaluator(embedding_dim=args.dis_embedding_dim, gru_units=args.dis_gru_units,
                              vocab_size=self.vocab_size, encoder_dim=2048)
        self.evaluator.to(self.device)
        self.gen_checkpoint_path = args.storage + '/ckpts/' + args.dataset + '/gen/' + args.gen_checkpoint_filename
        self.dis_checkpoint_path = args.storage + '/ckpts/' + args.dataset + '/dis/' + args.dis_checkpoint_filename
        self.evaluator_checkpoint_path = self.dis_checkpoint_path

        self.gen_optimizer = Optimizer(self.generator)
        self.dis_optimizer = optim.Adam(self.discriminator.parameters(), lr=args.dis_lr)  # 判别器优化
        self.dis_criterion = nn.BCELoss()  # 二分类交叉熵损失
        
        self.rl_criterion = losses.create(cfg.LOSSES.RL_TYPE).to(self.device)

    def setup_loader_train(self, epoch):
        self.training_loader = data_loader.load_train(
            self.distributed, epoch, self.set)
        

    def setup_loader_dis(self, epoch):
        self.set = dataset.Dataset(
            image_ids_path=cfg.DATA_LOADER.TRAIN_ID,
            input_seq=cfg.DATA_LOADER.INPUT_SEQ_PATH,
            target_seq=cfg.DATA_LOADER.TARGET_SEQ_PATH,
            gv_feat_path=cfg.DATA_LOADER.TRAIN_GV_FEAT,
            att_feats_folder=cfg.DATA_LOADER.TRAIN_ATT_FEATS,
            seq_per_img=cfg.DATA_LOADER.SEQ_PER_IMG,
            max_feat_num=cfg.DATA_LOADER.MAX_FEAT,
            dis='discriminator'
        )
        self.distraining_loader = data_loader.load_distrain(
            self.distributed, epoch, self.set)
        

    def make_kwargs(self, indices, input_seq, target_seq, gv_feat, att_feats, att_mask):
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


    def train(self):
        gen_batch_id = 0
        dis_batch_id = 0
        gen_epoch = 0
        dis_epoch = 0
        if path.isfile(self.gen_checkpoint_path):
            logging.info('loaded generator checkpoint')
            checkpoint = torch.load(self.gen_checkpoint_path)
            self.generator.load_state_dict(checkpoint['gen_state_dict'])
            self.generator.to(self.device)
            if args.gen_checkpoint_filename.split('_')[0] == 'PG':
                self.gen_optimizer.load_state_dict(checkpoint['gen_optimizer_state_dict'])
                gen_batch_id = checkpoint['gen_batch_id']
                gen_epoch = checkpoint['gen_epoch']

        if path.isfile(self.dis_checkpoint_path):
            logging.info('loaded discriminator checkpoint')
            checkpoint = torch.load(self.dis_checkpoint_path)
            self.discriminator.load_state_dict(checkpoint['dis_state_dict'])
            if args.dis_checkpoint_filename.split('_')[0] == 'PG':
                self.dis_optimizer.load_state_dict(checkpoint['dis_optimizer_state_dict'])
                dis_batch_id = checkpoint['dis_batch_id']
                dis_epoch = checkpoint['dis_epoch']
        
        if path.isfile(self.evaluator_checkpoint_path):
            logging.info('loaded evaluator checkpoint')
            checkpoint = torch.load(self.evaluator_checkpoint_path)
            self.evaluator.load_state_dict(checkpoint['dis_state_dict'])
        
        gen_pg_losses = AverageMeter()
        dis_losses = AverageMeter()
        dis_acc = AverageMeter()
        top5 = AverageMeter()
        top1 = AverageMeter()

        completed_epoch = False
        self.setup_loader_train(0)#epoch在数据加载时未起作用，因此设定一个固定值写死数据加载
        gen_iter = iter(self.training_loader)  # 生成迭代器
        self.setup_loader_dis(0)
        dis_iter = iter(self.distraining_loader)  # 生成迭代器
            
        for epoch in range(args.epochs):
            if gen_epoch == args.gen_epochs:
                break
            i = 0
            #print('##args.g_steps#########################################',args.g_steps)
            while i < args.g_steps:
                #print('i:',i)
                try:
                    start_time = time.time()
                    #print(epoch,len(next(gen_iter)))
                    indices, input_seq, target_seq, gv_feat, att_feats = next(gen_iter)
                    self.gen_train(indices=indices, input_seq=input_seq, target_seq=target_seq,gv_feat=gv_feat,att_feats=att_feats,
                              losses=gen_pg_losses,top1=top1,top5=top5)
                    time_taken = time.time() - start_time
                    if gen_batch_id % args.gen_print_freq == 0:
                        logging.info('GENERATOR: ADV EPOCH: [{}]\t'
                                     'GEN Epoch: [{}]\t'
                                     'GEN Batch: [{}]\t'
                                     'Time per batch: [{:.3f}]\t'
                                     'PG Loss [{:.4f}]({:.3f})\t'#策略梯度
                                     'Top 1 accuracy [{:.4f}]({:.3f})\t'
                                     'Top 5 accuracy [{:.4f}]({:.3f})'
                                     .format(epoch, gen_epoch, gen_batch_id, time_taken,gen_pg_losses.avg,gen_pg_losses.val,top1.avg,top1.val,top5.avg,top5.val))
                        if args.save_stats:
                            with open(args.storage + '/stats/' + args.dataset +
                                      '/gen/{}_'.format('TRAIN_PG_GEN') +
                                      'G-STEPS_{}_'.format(args.g_steps) +
                                      'D-STEPS_{}.csv'.format(args.d_steps), "a+") as file:
                                writer = csv.writer(file)
                                writer.writerow([epoch, gen_epoch, gen_batch_id, gen_pg_losses.avg,gen_pg_losses.val,top1.avg,top1.val,top5.avg,top5.val])
                    gen_batch_id += 1
                    i += 1
                except StopIteration:
                    #print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                    gen_batch_id = 0
                    gen_pg_losses.reset()
                    gen_epoch += 1
                    gen_iter = iter(self.training_loader)
                    completed_epoch = True
            i = 0
            #print('##args.d_steps#########################################',args.d_steps)
            while i < args.d_steps:
                
                try:
                    start_time = time.time()
                    indices, input_seq, target_seq, gv_feat, att_feats,  misgv_feat, misatt_feats = next(dis_iter)
                    self.dis_train(indices=indices, input_seq=input_seq, target_seq=target_seq,
                                   gv_feat=gv_feat, att_feats=att_feats,
                                   misgv_feat=misgv_feat, misatt_feats=misatt_feats,
                              losses=dis_losses, acc=dis_acc)
                    time_taken = time.time() - start_time
                    if dis_batch_id % args.dis_print_freq == 0:
                        logging.info('DISCRIMINATOR: ADV Epoch: [{}]\t'
                                     'DIS Epoch: [{}]\t'
                                     'DIS Batch: [{}]\t'
                                     'Time per batch: [{:.3f}]\t'
                                     'Loss [{:.4f}]({:.3f})\t'
                                     'Accuracy [{:.4f}]({:.3f})'.format(epoch, dis_epoch, dis_batch_id, time_taken,
                                                                        dis_losses.avg, dis_losses.val, dis_acc.avg,dis_acc.val))
                        if args.save_stats:
                            with open(args.storage + '/stats/' + args.dataset +
                                      '/dis/{}_'.format('TRAIN_PG_DIS') +
                                      'LR_{}_'.format(args.dis_lr) +
                                      'G-STEPS_{}_'.format(args.g_steps) +
                                      'D-STEPS_{}.csv'.format(args.d_steps), 'a+') as file:
                                writer = csv.writer(file)
                                writer.writerow([epoch, gen_epoch, gen_batch_id, dis_epoch, dis_batch_id, dis_losses.avg, dis_losses.val, dis_acc.avg,dis_acc.val])
                    dis_batch_id += 1
                    i += 1
                except StopIteration:
                    dis_losses.reset()
                    dis_acc.reset()
                    dis_epoch += 1
                    dis_batch_id = 0
                    dis_iter = iter(self.distraining_loader)

            if epoch % args.val_freq == 0 or completed_epoch:
                val=self.validate(epoch,gen_epoch,gen_batch_id)
                if args.save_models:
                    torch.save(
                        {'gen_state_dict': self.generator.state_dict(), 'optimizer_state_dict': self.gen_optimizer.state_dict(),
                         'gen_batch_id': gen_batch_id, 'gen_epoch': gen_epoch}, args.storage + '/ckpts/' + args.dataset +
                                                                                '/gen/{}_'.format('TRAIN_PG_GEN') +
                                                                                'G-STEPS_{}_'.format(args.g_steps) +
                                                                                'D-STEPS_{}_'.format(args.d_steps) +
                                                                                '{}_'.format(epoch) +
                                                                                '{}_'.format(gen_epoch) +
                                                                                '{}.pth'.format(gen_batch_id))
                    torch.save(
                        {'dis_state_dict': self.discriminator.state_dict(), 'optimizer_state_dict': self.dis_optimizer.state_dict(),
                         'dis_batch_id': dis_batch_id, 'dis_epoch': dis_epoch}, args.storage + '/ckpts/' + args.dataset +
                                                                                '/dis/{}_'.format('TRAIN_PG_DIS') +
                                                                                'G-STEPS_{}_'.format(args.g_steps) +
                                                                                'D-STEPS_{}_'.format(args.d_steps) +
                                                                                '{}_'.format(epoch) +
                                                                                '{}_'.format(dis_epoch) +
                                                                                '{}.pth'.format(dis_batch_id))
                completed_epoch = False
            self.gen_optimizer.scheduler_step('Epoch', val)
            self.scheduled_sampling(epoch)



    def forward(self, kwargs,max_len):

        ids = kwargs[cfg.PARAM.INDICES]
        gv_feat = kwargs[cfg.PARAM.GLOBAL_FEAT]
        att_feats = kwargs[cfg.PARAM.ATT_FEATS]
        att_mask = kwargs[cfg.PARAM.ATT_FEATS_MASK]

        # max贪婪解码下计算奖励，分别为判别器和语言评估器的奖励和作为基线
        kwargs['BEAM_SIZE'] = 1
        kwargs['GREEDY_DECODE'] = True
        kwargs[cfg.PARAM.GLOBAL_FEAT] = gv_feat
        kwargs[cfg.PARAM.ATT_FEATS] = att_feats
        kwargs[cfg.PARAM.ATT_FEATS_MASK] = att_mask

        self.generator.eval()
        with torch.no_grad():
            seq_max, logP_max = self.generator.decode(**kwargs)
        #判别器的奖励
        self.discriminator.eval()
        rewards_max_dis = self.discriminator(kwargs[cfg.PARAM.ATT_FEATS], seq_max, max_len)
        rewards_max_dis = utils.expand_numpy(rewards_max_dis).cpu().detach().numpy()
        self.generator.train()
        #语言评估的奖励
        rewards_max, rewards_info_max = self.scorer(ids, seq_max.data.cpu().numpy().tolist())
        rewards_max = utils.expand_numpy(rewards_max)
        #rewards_max = self.evaluator(kwargs[cfg.PARAM.ATT_FEATS], seq_max, max_len)
        #rewards_max = utils.expand_numpy(rewards_max).cpu().detach().numpy()
        #按比例计算奖励和作为基线
        rewards_m=args.lamda*rewards_max_dis+(1-args.lamda)*rewards_max
        #print('rewards_max_dis',rewards_max_dis)
        #print('rewards_max',rewards_max)
        #print('rewards_m',rewards_m)

        ids = utils.expand_numpy(ids)
        gv_feat = utils.expand_tensor(gv_feat, cfg.DATA_LOADER.SEQ_PER_IMG)
        att_feats = utils.expand_tensor(att_feats, cfg.DATA_LOADER.SEQ_PER_IMG)
        att_mask = utils.expand_tensor(att_mask, cfg.DATA_LOADER.SEQ_PER_IMG)

        # sample生成器随机采样生成奖励
        kwargs['BEAM_SIZE'] = 1
        kwargs['GREEDY_DECODE'] = False
        kwargs[cfg.PARAM.GLOBAL_FEAT] = gv_feat
        kwargs[cfg.PARAM.ATT_FEATS] = att_feats
        kwargs[cfg.PARAM.ATT_FEATS_MASK] = att_mask

        seq_sample, logP_sample = self.generator.decode(**kwargs)
        
        #判别器奖励
        rewards_sample_dis = self.discriminator(kwargs[cfg.PARAM.ATT_FEATS], seq_sample, max_len)
        rewards_sample_dis = utils.expand_numpy(rewards_sample_dis).cpu().detach().numpy()
        #语言评估奖励
        rewards_sample, rewards_info_sample = self.scorer(ids, seq_sample.data.cpu().numpy().tolist())
        #rewards_sample = self.evaluator(kwargs[cfg.PARAM.ATT_FEATS], seq_sample, max_len)
        #rewards_sample = utils.expand_numpy(rewards_sample).cpu().detach().numpy()
        #求和
        rewards_s=args.lamda*rewards_sample_dis+(1-args.lamda)*rewards_sample
        #print('rewards_sample_dis',rewards_sample_dis)
        #print('rewards_sample',rewards_sample)
        #print('rewards_s',rewards_s)
        #最终奖励为生成器生成样本的奖励减去基线
        rewards = rewards_s - rewards_m
        #print('rewards',rewards)
        #print('logP_sample',logP_sample)
        rewards = torch.from_numpy(rewards).float().cuda()
        self.gen_optimizer.zero_grad()
        #最终损失为奖励与采样生成句子的概率相乘
        loss = self.rl_criterion(seq_sample, logP_sample, rewards)
        '''
        loss_info = {}
        for key in rewards_info_sample:
            loss_info[key + '_sample'] = rewards_info_sample[key]
        for key in rewards_info_max:
            loss_info[key + '_max'] = rewards_info_max[key]
        '''
        return loss

    def gen_train(self,indices, input_seq, target_seq,gv_feat,att_feats,losses,top1,top5):
        self.discriminator.eval()
        self.generator.eval()
        self.evaluator.eval()

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
        #batch_size = gv_feat.size(0)
        kwargs, max_len = self.make_kwargs(indices, input_seq, target_seq, gv_feat, att_feats, att_mask)           
        preds = self.generator(**kwargs).to(self.device)
        loss = self.forward(kwargs,max_len)
        Preds = preds.view(-1, preds.shape[-1])
        Target_seq = kwargs[cfg.PARAM.TARGET_SENT].view(-1)

        #fake_caps,state = self.generator(**kwargs).to(self.device)
        #fake_caps = F.avg_pool1d(fake_caps, fake_caps.shape[-1]).squeeze(-1)
        #fake_caps = fake_caps.view(batch_size, cfg.DATA_LOADER.SEQ_PER_IMG * max_len)
        #fake_preds = self.evaluator(kwargs[cfg.PARAM.ATT_FEATS], fake_caps, max_len)
        
        loss.backward()
        utils.clip_gradient(self.gen_optimizer.optimizer, self.generator,
                            cfg.SOLVER.GRAD_CLIP_TYPE, cfg.SOLVER.GRAD_CLIP)
        
        self.gen_optimizer.step()
        
        #self.gen_optimizer.scheduler_step('Iter')
        top1_acc = categorical_accuracy(Preds, Target_seq, 1)
        # print(top1_acc)
        top5_acc = categorical_accuracy(Preds, Target_seq, 5)
        # print(top5_acc)
        top1.update(top1_acc)
        top5.update(top5_acc)
        losses.update(loss.item())

    def dis_train(self,indices, input_seq, target_seq, gv_feat, att_feats,misgv_feat, misatt_feats,
                  losses, acc):
        self.discriminator.train()
        self.generator.eval()

        atts_num = [x.shape[0] for x in att_feats]
        max_att_num = np.max(atts_num)
        mask_arr = []
        for i, num in enumerate(atts_num):
            tmp_mask = np.zeros((1, max_att_num), dtype=np.float32)
            tmp_mask[:, 0:num] = 1
            mask_arr.append(torch.from_numpy(tmp_mask))
        att_mask = torch.cat(mask_arr, 0)

        misatts_num = [x.shape[0] for x in misatt_feats]
        max_misatt_num = np.max(misatts_num)
        mismask_arr = []
        for i, num in enumerate(misatts_num):
            tmp_mask = np.zeros((1, max_misatt_num), dtype=np.float32)
            tmp_mask[:, 0:num] = 1
            mismask_arr.append(torch.from_numpy(tmp_mask))
        misatt_mask = torch.cat(mismask_arr, 0)

        input_seq = input_seq.to(self.device)
        target_seq = target_seq.to(self.device)
        gv_feat = gv_feat.to(self.device)
        att_feats = att_feats.to(self.device)
        att_mask = att_mask.to(self.device)
        misgv_feat = misgv_feat.to(self.device)
        misatt_feats = misatt_feats.to(self.device)
        misatt_mask = misatt_mask.to(self.device)

        batch_size = gv_feat.size(0)
        kwargs, max_len = self.make_kwargs(indices, input_seq, target_seq, gv_feat, att_feats, att_mask)
        kwargs1, max_len = self.make_kwargs(indices, input_seq, target_seq, misgv_feat, misatt_feats, misatt_mask)
        kwargs['GREEDY_DECODE'] = cfg.INFERENCE.GREEDY_DECODE
        #kwargs['BEAM_SIZE'] = cfg.INFERENCE.BEAM_SIZE
        # seq, _ = model.decode(**kwargs)
        fake_caps, _ = self.generator.decode(**kwargs)
        # print(fake_caps)
        # fake_caps = F.avg_pool1d(fake_caps, fake_caps.shape[-1]).squeeze(-1)
        Target_seq = kwargs[cfg.PARAM.TARGET_SENT]
        # print(Target_seq)
        Target_seq = Target_seq.ne(-1).long() * Target_seq
        # print(Target_seq)
        Target_seq = Target_seq.view(batch_size, cfg.DATA_LOADER.SEQ_PER_IMG * max_len)
        # fake_caps=fake_caps.view(batch_size,cfg.DATA_LOADER.SEQ_PER_IMG*max_len)

        self.dis_optimizer.zero_grad()  # 梯度初始化为0

        true_preds = self.discriminator(kwargs[cfg.PARAM.ATT_FEATS], Target_seq, max_len)
        false_preds = self.discriminator(kwargs1[cfg.PARAM.ATT_FEATS], Target_seq, max_len)
        fake_preds = self.discriminator(kwargs[cfg.PARAM.ATT_FEATS], fake_caps, max_len)

        a = [1 for i in range(batch_size)]
        b = [0 for i in range(batch_size)]
        ones = torch.tensor(a).float()
        zeros = torch.tensor(b).float()

        loss = self.dis_criterion(true_preds.cpu(), ones) + 0.5 * self.dis_criterion(false_preds.cpu(), zeros) + \
               0.5 * self.dis_criterion(fake_preds.cpu(), zeros)
        loss.backward()
 
        self.dis_optimizer.step()
        losses.update(loss.item())
        true_acc = binary_accuracy(true_preds.cpu(), ones).item()
        false_acc = binary_accuracy(false_preds.cpu(), zeros).item()
        fake_acc = binary_accuracy(fake_preds.cpu(), zeros).item()
        avg_acc = (true_acc + false_acc + fake_acc) / 3.0
        acc.update(avg_acc)

    def validate(self,epoch, gen_epoch, gen_batch_id):

        val_res = self.val_evaler(self.generator, 'val_' + str(epoch))
        
        #logging.info(str(val_res))
        logging.info('VALIDATION')
        logging.info('ADV Epoch: [{}]\t'
                     'GEN Epoch: [{}]\t'
                     'GEN Batch: [{}]\t'
                     'metrics:[{}]\t'
                     .format(epoch, gen_epoch, gen_batch_id,str(val_res)))
        if args.save_stats:
            with open(args.storage + '/stats/' + args.dataset +
                      '/gen/{}_'.format('VAL_PG_GEN') +
                      'G-STEPS_{}_'.format(args.g_steps) +
                      'D-STEPS_{}.csv'.format(args.d_steps), 'a+') as file:
                writer = csv.writer(file)
                writer.writerow([epoch, gen_epoch, gen_batch_id,str(val_res)])
        val = 0
        for score_type, weight in zip(cfg.SCORER.TYPES, cfg.SCORER.WEIGHTS):
            val -= val_res[score_type] * weight
        return val


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Adversarial Training via Policy Gradients')
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--gen-epochs', type=int, default=5)
    parser.add_argument('--g-steps', type=int, default=1)
    parser.add_argument('--d-steps', type=int, default=1)
    parser.add_argument('--dis-lr', type=float, default=1e-5)
    parser.add_argument('--lamda', type=float, default=0)#分别控制比例
    parser.add_argument('--val-freq', type=int, default=5)
    parser.add_argument('--gen-print-freq', type=int, default=5)
    parser.add_argument('--dis-print-freq', type=int, default=5)
    parser.add_argument('--save-stats', type=bool, default=True)
    parser.add_argument('--save-models', type=bool, default=True)
    parser.add_argument('--storage', type=str, default='.')
    #parser.add_argument('--dataset', type=str, default='CXR')
    parser.add_argument('--dataset', type=str, default='LGK')
    parser.add_argument('--dis-embedding-dim', type=int, default=512)
    parser.add_argument('--dis-gru-units', type=int, default=512)
    parser.add_argument('--gen-checkpoint-filename', type=str, default='MLE_GEN_0.pth')
    parser.add_argument('--dis-checkpoint-filename', type=str, default='PRETRAIN_DIS_0.pth')
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--folder', dest='folder', type=str, default='./')
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--result-folder", type=str, default='./stats/LGK/gen/pg/result')
    
    
    args = parser.parse_args()
    cfg_from_file(os.path.join(args.folder, 'configpg.yml'))
    #cfg_from_file(os.path.join(args.folder, 'configtranspg.yml'))
    trainer = Trainer(args)
    trainer.train()
