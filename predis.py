import argparse
import json
import os
import torch.distributed as dist
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence
import os.path as path
import logging
import time
import csv
from Datasets import dataset, data_loader
import model
from datasets import ImageCaptionDataset
from models import *
from lib.utils import *
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

#logging.info(torch.cuda.is_available())
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

logging.basicConfig(level=logging.INFO)
from lib.config import cfg, cfg_from_file


class predis(object):
    def __init__(self, args):
        super(predis, self).__init__()
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #self.device = torch.device("cpu")
        self.vocab_size = cfg.MODEL.VOCAB_SIZE + 1
        self.num_gpus = 1
        self.distributed = self.num_gpus > 1
        if self.distributed:
            torch.cuda.set_device(args.local_rank)
            torch.distributed.init_process_group(
                backend="nccl", init_method="env://"
            )
        self.setup_network()


    def setup_network(self):
        g=model.create(cfg.MODEL.TYPE)
        if self.distributed:
            # this should be removed if we update BatchNorm stats
            self.generator = torch.nn.parallel.DistributedDataParallel(
                self.generator.to(self.device),
                device_ids = [self.args.local_rank],
                output_device = self.args.local_rank,
                broadcast_buffers = False
            )
        else:
            self.generator=g.to(self.device)
        self.discriminator = GRUDiscriminator(embedding_dim=args.dis_embedding_dim,
                                              gru_units=args.dis_gru_units,
                                              vocab_size=self.vocab_size,
                                              encoder_dim=2048)
        self.discriminator.to(self.device)
        gen_checkpoint_path = args.storage + '/ckpts/' + args.dataset + '/gen/' + args.gen_checkpoint_filename
        dis_checkpoint_path = args.storage + '/ckpts/' + args.dataset + '/dis/' + args.dis_checkpoint_filename

        if path.isfile(gen_checkpoint_path):
            checkpoint = torch.load(gen_checkpoint_path)
            self.generator.load_state_dict(checkpoint['gen_state_dict'])
            logging.info('loaded generator')
            self.generator.to(self.device)

        if path.isfile(dis_checkpoint_path):
            checkpoint = torch.load(dis_checkpoint_path)
            self.discriminator.load_state_dict(checkpoint['dis_state_dict'])
            self.dis_optimizer.load_state_dict(checkpoint['dis_optimizer_state_dict'])
            logging.info('loaded discriminator')
            self.discriminator.to(self.device)

        self.dis_optimizer = optim.Adam(self.discriminator.parameters(), lr=args.lr)  # 判别器优化
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.dis_optimizer, step_size=args.step_size,
                                                         gamma=0.5)  # 调整学习率
        self.dis_criterion = nn.BCELoss()  # 二分类交叉熵损失


    def setup_loader_train(self,  epoch):
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
        self.training_loader = data_loader.load_distrain(
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

    def train(self):

        for e in range(args.epochs):
            self.distrain(e)
            if args.save_model:
                torch.save(
                    {'dis_state_dict': self.discriminator.state_dict(), 'optimizer_state_dict': self.dis_optimizer.state_dict()},
                    args.storage + '/ckpts/' + args.dataset +
                    '/dis/{}_{}.pth'.format('PRETRAIN_DIS', e))
            logging.info('Completed epoch: ' + str(e))
            self.scheduler.step()

    def distrain(self,epoch):
        losses = AverageMeter()
        acc = AverageMeter()

        self.discriminator.train()
        self.generator.eval()
        self.setup_loader_train(epoch)

        for batch_id, (indices, input_seq, target_seq, gv_feat, att_feats, misgv_feat, misatt_feats) in enumerate(self.training_loader):
            start_time = time.time()

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
            misgv_feat=misgv_feat.to(self.device)
            misatt_feats=misatt_feats.to(self.device)
            misatt_mask=misatt_mask.to(self.device)

            batch_size = gv_feat.size(0)
            kwargs, max_len = self.make_kwargs(indices, input_seq, target_seq, gv_feat, att_feats, att_mask)
            kwargs1, max_len = self.make_kwargs(indices, input_seq, target_seq, misgv_feat, misatt_feats, misatt_mask)
            kwargs['GREEDY_DECODE'] = cfg.INFERENCE.GREEDY_DECODE
            #kwargs['BEAM_SIZE'] = cfg.INFERENCE.BEAM_SIZE
            #seq, _ = model.decode(**kwargs)
            fake_caps,_ = self.generator.decode(**kwargs)
            #print(fake_caps)
            #fake_caps = F.avg_pool1d(fake_caps, fake_caps.shape[-1]).squeeze(-1)
            Target_seq = kwargs[cfg.PARAM.TARGET_SENT]
            #print(Target_seq)
            Target_seq = Target_seq.ne(-1).long() * Target_seq
            #print(Target_seq)
            Target_seq=Target_seq.view(batch_size,cfg.DATA_LOADER.SEQ_PER_IMG*max_len)
            #fake_caps=fake_caps.view(batch_size,cfg.DATA_LOADER.SEQ_PER_IMG*max_len)


            self.dis_optimizer.zero_grad() #梯度初始化为0

            true_preds = self.discriminator(kwargs[cfg.PARAM.ATT_FEATS], Target_seq, max_len)
            false_preds = self.discriminator(kwargs1[cfg.PARAM.ATT_FEATS], Target_seq, max_len)
            fake_preds = self.discriminator(kwargs[cfg.PARAM.ATT_FEATS], fake_caps, max_len)

            a = [1 for i in range(batch_size)]
            b=[0 for i in range(batch_size)]
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

            if batch_id % args.print_freq == 0:
                logging.info('Epoch: [{}]\t'
                             'Batch: [{}]\t'
                             'Time per batch: [{:.3f}]\t'
                             'Loss [{:.4f}]\t'
                             'Accuracy [{:.4f}]'.format(epoch, batch_id, time.time() - start_time, losses.avg,
                                                                acc.avg))

                if args.save_stats:
                    with open(args.storage + '/stats/' + args.dataset +
                              '/dis/{}.csv'.format('PRETRAIN_DIS'), 'a+') as file:
                        writer = csv.writer(file)
                        writer.writerow(
                            [epoch, batch_id, losses.avg, acc.avg])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pre-train discriminator')

    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--step-size', type=float, default=5)
    parser.add_argument('--print-freq', type=int, default=50)

    #parser.add_argument('--max-len', type=int, default=10)
    #parser.add_argument('--max-len', type=int, default=25)
    parser.add_argument('--storage', type=str, default='.')
    #parser.add_argument('--image-path', type=str, default='images')
    #parser.add_argument('--dataset', type=str, default='CXR')
    parser.add_argument('--dataset', type=str, default='LGK')
    parser.add_argument('--dis-embedding-dim', type=int, default=512)
    parser.add_argument('--dis-gru-units', type=int, default=512)
    #parser.add_argument('--gen-embedding-dim', type=int, default=512)
    #parser.add_argument('--gen-gru-units', type=int, default=512)
    #parser.add_argument('--attention-dim', type=int, default=512)

    parser.add_argument('--gen-checkpoint-filename', type=str, default='MLE_GEN_0.pth')
    parser.add_argument('--dis-checkpoint-filename', type=str, default='')

    parser.add_argument('--save-model', type=bool, default=True)
    parser.add_argument('--save-stats', type=bool, default=True)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--folder', dest='folder', type=str, default='./')
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()
    cfg_from_file(os.path.join(args.folder, 'configmle.yml'))
    predis = predis(args)
    predis.train()
