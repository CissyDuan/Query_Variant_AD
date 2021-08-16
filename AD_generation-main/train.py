from __future__ import division
from __future__ import print_function

import time
import argparse
import torch
import torch.nn as nn
from optims import Optim
import util
from util import utils
import lr_scheduler as L
from models import *
from collections import OrderedDict
from tqdm import tqdm
import sys
import os
import Data
import file
import metric



parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
current_dir = os.getcwd()
sys.path.insert(0, parent_dir)
from util.nlp_utils import *


# config
def parse_args():
    parser = argparse.ArgumentParser(description='train.py')
    parser.add_argument('-config', default='config.yaml', type=str,
                        help="config file")
    parser.add_argument('-model', default='transformer_gcn', type=str,
                        choices=['seq2seq',  'transformer','transformer_gcn'])
    parser.add_argument('-gpus', default=[1], type=int,
                        help="Use CUDA on the listed devices.")
    parser.add_argument('-restore',
                        type=str, default='',
                        help="restore checkpoint")
    parser.add_argument('-seed', type=int, default=2222,
                        help="Random seed")
    parser.add_argument('-type', default='train', choices=['train', 'eval'],
                        help='train type or eval')
    parser.add_argument('-log', default='', type=str,
                        help="log directory")
    parser.add_argument('-train_type', default='generate', type=str,
                        choices=['generate','sample_rl'])


    opt = parser.parse_args()
    # 用config.data来得到config中的data选项
    config = util.utils.read_config(opt.config)
    return opt, config


# set opt and config as global variables
args, config = parse_args()
random.seed(args.seed)
np.random.seed(args.seed)


# Training settings

def set_up_logging():
    # log为记录文件
    # config.log是记录的文件夹, 最后一定是/
    # opt.log是此次运行时记录的文件夹的名字
    if not os.path.exists(config.log):
        os.mkdir(config.log)
    if args.log == '':
        log_path = config.log + utils.format_time(time.localtime()) + '/'
    else:
        log_path = config.log + args.log + '/'
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    logging = utils.logging(log_path + 'log.txt')  # 往这个文件里写记录
    logging_csv = utils.logging_csv(log_path + 'record.csv')  # 往这个文件里写记录
    for k, v in config.items():
        logging("%s:\t%s\n" % (str(k), str(v)))
    logging("\n")
    return logging, logging_csv, log_path


logging, logging_csv, log_path = set_up_logging()
use_cuda = torch.cuda.is_available()



def train(model_sample,model_generate, vocab, dataloader_train, dataloader_dev,dataloader_dev_qk,scheduler_sample,scheduler_generate,optim_sample,optim_generate, updates):
    scores = []
    max_bleu = 0.
    for epoch in range(1, config.epoch + 1):
        total_acc = 0.
        total_loss=0
        start_time = time.time()

        model_sample.train()
        model_generate.train()


        if config.schedule:
            scheduler_sample.step()
            scheduler_generate.step()
            print("Decaying sample learning rate to %g" % scheduler_sample.get_lr()[0])
            print("Decaying generate learning rate to %g" % scheduler_generate.get_lr()[0])

        loss_view=0
        prob_view=0
        batch_view=0
        loss_rl_view=0

        for batch in tqdm(dataloader_train):
            if len(batch.sub_node_idx)==0:
                continue

            model_sample.zero_grad()
            model_generate.zero_grad()

            sample_node_idx, sample_adj, sample_adj_weight, sample_prob, tgt, word_type, query_batch, sub_batch = model_sample(
                    batch, train_type=args.train_type, sample=True, sample_type='train')

            assert args.train_type in ['generate', 'sample_rl']
            if args.train_type=='generate':
                output, x, state = model_generate(sample_node_idx, sample_adj, sample_adj_weight, tgt, word_type)
                generate_loss, acc = model_generate.compute_loss(output, tgt)
                if torch.isnan(generate_loss):
                    break
                generate_loss.backward()

                optim_generate.step()
                sample_prob = [s.cuda() for s in sample_prob]
                sample_prob = torch.stack(sample_prob, 0)
                prob_num = sample_prob.size(0)

                updates += 1

                total_loss += generate_loss.data.item()
                loss_view += generate_loss.data.item()
                prob_view += ((sample_prob.sum()) / prob_num).data
                batch_view += 1
                if batch_view == 100:
                    logging('loss {} prob {} epoch {} update {} \n'.format(loss_view, prob_view, epoch, updates))
                    loss_view = 0
                    prob_view = 0
                    batch_view = 0
                total_acc += acc


            elif args.train_type=='sample_rl':

                output, x, state = model_generate(sample_node_idx, sample_adj, sample_adj_weight,tgt, word_type)
                generate_loss, acc = model_generate.compute_loss_sample(output, tgt)
                loss_view_batch=generate_loss.mean()

                sample_prob = [s.cuda() for s in sample_prob]
                sample_prob_ori= torch.stack(sample_prob, 0)
                sample_prob = torch.sigmoid(sample_prob_ori - 1)
                sample_prob=-torch.log(sample_prob)

                prob_num = sample_prob.size(0)
                generate_loss = torch.tanh(generate_loss)
                generate_loss=generate_loss.detach().data

                reward=1-generate_loss


                loss = torch.mul(reward,sample_prob)
                loss = (loss.sum(0))/ prob_num

                loss.backward()

                optim_sample.step()

                updates += 1


                prob_view += ((sample_prob_ori.sum()) / prob_num).data
                total_loss += loss_view_batch.data
                loss_view += loss_view_batch.data
                loss_rl_view+=loss.data

                batch_view += 1
                if batch_view == 100:
                    logging('loss {} prob {} loss_rl{}  epoch {} update {} \n'.format(loss_view,prob_view,loss_rl_view,epoch, updates))
                    loss_view = 0
                    prob_view = 0
                    batch_view = 0
                    loss_rl_view=0
                total_acc += acc

            else:
                print('error')
                break



            if updates % config.eval_interval == 0:
                logging("time: %6.3f, epoch: %3d, updates: %8d, train loss: %6.3f, train acc: %.3f\n"
                        % (time.time() - start_time, epoch, updates, total_loss / config.eval_interval,
                           total_acc / config.eval_interval))
                logging("learning rate to %g" % scheduler_generate.get_lr()[0])
                print('evaluating after %d updates...\r' % updates)
                # TODO: fix eval and print bleu, ppl
                score = eval(model_sample,model_generate, vocab, dataloader_dev,dataloader_dev_qk, epoch, updates)
                scores.append(score)
                if score >= max_bleu:
                    save_model(log_path + str(score) + '_checkpoint.pt', model_sample,model_generate,optim_sample, optim_generate, updates)
                    max_bleu = score

                model_sample.train()
                model_generate.train()
                total_loss = 0.
                total_acc = 0.
                start_time = time.time()

            if updates % config.save_interval == 0:
                save_model(log_path + str(updates) + '_updates_checkpoint.pt', model_sample,model_generate, optim_sample,optim_generate, updates)
    dataloader_test = get_dataloader(vocab, split='test', train_type='k')
    dataloader_test_qk = get_dataloader(vocab, split='test', train_type='qk')

    eval(model_sample, model_generate, vocab, dataloader_test, dataloader_test_qk, 0, updates)
    return max_bleu



def eval(model_sample,model_generate, vocab, dataloader_k, dataloader_qk,epoch, updates):
    model_sample.eval()
    model_generate.eval()

    multi_ref,query,keyword,keyword_ng=[],[],[],[]
    candidate,candidate_s,candidate_qk, candidate_qk_s= [], [], [],[]
    expose_sum=[]

    for batch in tqdm(dataloader_k):

        sample_node_idx, sample_adj, sample_adj_weight, sample_prob, tgt, word_type,query_batch,sub_node_idx_batch= model_sample(batch, train_type=args.train_type,
                                                                                                   sample=False,
                                                                                                   sample_type='eval')
        expose=batch.expose
        expose=[int(e) for e in expose]
        expose_sum+=expose

        sub_batch_sent = [vocab.id2sent(k) for k in sub_node_idx_batch]
        query_batch_sent = [vocab.id2sent(s) for s in query_batch]
        ref = [vocab.id2sent(t[1:]) for t in tgt]

        multi_ref += ref
        keyword+=sub_batch_sent
        query+=query_batch_sent

        samples = model_generate.sample(sample_node_idx, sample_adj, sample_adj_weight, word_type)
        cand=[vocab.id2sent(s) for s in samples]
        candidate += cand

        sample_node_idx, sample_adj,sample_adj_weight, sample_prob, tgt, word_type ,query_batch_s,sub_node_idx_batch_s= model_sample(batch, train_type=args.train_type,sample=True,sample_type='eval')
        samples_s = model_generate.sample(sample_node_idx, sample_adj, sample_adj_weight,word_type)
        cand_s=[vocab.id2sent(s) for s in samples_s]
        candidate_s += cand_s

    for batch in tqdm(dataloader_qk):

        sample_node_idx, sample_adj, sample_adj_weight, sample_prob, tgt, word_type, query_batch_qk, sub_node_idx_batch_qk = model_sample(
            batch, train_type=args.train_type,
            sample=False,
            sample_type='eval')
        key = [vocab.id2sent(k) for k in sub_node_idx_batch_qk]
        '''for i in key:
            print(i)
        a = 1
        assert a == 0'''
        keyword_ng += key

        samples_qk = model_generate.sample(sample_node_idx, sample_adj, sample_adj_weight, word_type)
        cand_qk = [vocab.id2sent(s) for s in samples_qk]
        candidate_qk += cand_qk

        sample_node_idx, sample_adj, sample_adj_weight, sample_prob, tgt, word_type, query_batch, sub_node_idx_batch = model_sample(
            batch, train_type=args.train_type,
            sample=True,
            sample_type='eval')
        samples_qk_s = model_generate.sample(sample_node_idx, sample_adj, sample_adj_weight, word_type)
        cand_qk_s = [vocab.id2sent(s) for s in samples_qk_s]
        candidate_qk_s += cand_qk_s


    text_result, bleu = utils.eval_bleu(multi_ref, candidate, log_path)
    text_result_s, bleu_s = utils.eval_bleu(multi_ref, candidate_s, log_path)
    text_result_qk, bleu_qk = utils.eval_bleu(multi_ref, candidate_qk, log_path)
    text_result_qk_s, bleu_qk_s = utils.eval_bleu(multi_ref, candidate_qk_s, log_path)


    logging_csv([epoch, updates, text_result,text_result_s,text_result_qk,text_result_qk_s])

    print_list=[query,keyword,multi_ref, candidate,candidate_s,candidate_qk,candidate_qk_s,expose_sum]

    with open(log_path +"test_ppl.json", "w") as f:
        json.dump(print_list, f)
    print_list = [query, keyword, multi_ref, candidate, candidate_s, candidate_qk, candidate_qk_s]
    utils.write_result_to_file(print_list, log_path)

    candidate_list = [multi_ref, candidate, candidate_s, candidate_qk, candidate_qk_s]
    name = ['ori', 'nosample', 'sample', 'nosample+q', 'sample+q']
    bleu_target = 0
    for c, n in zip(candidate_list, name):
        bleu, recall_q, recall_k, recall_qk, dist1, dist2 = metric.get_metric(c, multi_ref, keyword, query, keyword_ng,
                                                                              expose_sum, log_path)
        logging(
            '{}: bleu {}, recall_q {}, recall_k {}, recall_qk {}, dist1 {}, dist2 {} \n'.format(n, bleu, recall_q, recall_k,
                                                                                          recall_qk, dist1, dist2))
        logging_csv([epoch, updates, n, bleu, recall_q, recall_k, recall_qk, dist1, dist2])
        if n == 'sample':
            bleu_target = bleu


    return bleu_target


def save_model(path, model_sample,model_generate, optim_sample,optim_generate, updates):

    model_state_dict_sample = model_sample.module.state_dict() if len(args.gpus) > 1 else model_sample.state_dict()
    model_state_dict_generate = model_generate.module.state_dict() if len(args.gpus) > 1 else model_generate.state_dict()
    checkpoints = {
        'model_sample': model_state_dict_sample,
        'model_generate': model_state_dict_generate,
        'config': config,
        'optim_sample': optim_sample,
        'optim_generate': optim_generate,
        'updates': updates}
    torch.save(checkpoints, path)

def get_dataloader(vocab,split=None,train_type=None):
    assert split in ['train','dev','test','variant']

    if split=='train':
        dataloader = DataLoader(config, config.data + split+'.json', config.batch_size, vocab,data_type='train',train_type=train_type)
    else:
        dataloader = DataLoader(config, config.data + split + '.json', config.batch_size, vocab, data_type='eval',train_type=train_type)


    return dataloader()


def main():
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)

    vocab=Vocab(config.vocab_file, config.emb_size, use_pre_emb=False, vocab_size=config.vocab_size)
    print('vocab clear')


    torch.backends.cudnn.benchmark = True

    # model
    print('building model...\n')
    # configure the model
    # Model and optimizer


    if args.model == 'seq2seq':
        model_generate= seq2seq(config, vocab, use_cuda, pretrain=None)
    elif args.model == 'transformer':
        model_generate = Transformer(config, vocab, use_cuda,pretrain=None)
    elif args.model == 'transformer_gcn':
        model_generate = Transformer_gcn(config, vocab, use_cuda,pretrain=None)




    if args.restore:
        print('loading checkpoint...\n')
        checkpoints = torch.load(os.path.join(log_path, args.restore))
        model_generate.load_state_dict(checkpoints['model_generate'])
        if args.train_type=='sample_rl':
            emb=model_generate.embedding
            model_sample = graph2seq_rl(config, vocab, use_cuda, emb, pretrain=None)
        elif args.train_type=='generate':
            model_sample = graph2seq_rl(config, vocab, use_cuda,0, pretrain=None)
            model_sample.load_state_dict(checkpoints['model_sample'])
        else:
            print('err')
    else:
        #model_sample = graph2seq_rl(config, vocab, use_cuda, model_generate.embedding, pretrain=None)
        model_sample = graph2seq_rl(config, vocab, use_cuda, 0, pretrain=None)

    '''if args.restore:
        print('loading checkpoint...\n')
        checkpoints = torch.load(os.path.join(log_path, args.restore))

        model_sample.load_state_dict(checkpoints['model_sample'])
        model_generate.load_state_dict(checkpoints['model_generate'])'''

    if use_cuda:
        model_sample.cuda()
        model_generate.cuda()

    # if len(args.gpus) > 1:  # 并行
    # model = nn.DataParallel(model, device_ids=args.gpus, dim=1)
    logging(repr(model_sample) + "\n\n")  # 记录这个文件的框架
    logging(repr(model_generate) + "\n\n")

    # total number of parameters
    sample_param_count = 0
    generate_param_count=0

    for param in model_sample.parameters():
        sample_param_count += param.view(-1).size()[0]
    for param in model_generate.parameters():
        generate_param_count += param.view(-1).size()[0]

    logging('total number of sample parameters: %d\n\n' % sample_param_count)
    logging('total number of generate parameters: %d\n\n' % generate_param_count)

    print('# generator parameters:', sum(param.numel() for param in model_generate.parameters()))

    # updates是已经进行了几个epoch, 防止中间出现程序中断的情况.
    if args.restore:
        updates = checkpoints['updates']
        ori_updates = updates
    else:
        updates = 0

    # optimizer
    '''if args.restore:
        optim_sample = checkpoints['optim_sample']
        optim_generate = checkpoints['optim_generate']
    else:'''
    #optimizer = optim.Adam(self.params, lr=self.lr)
    optim_sample = Optim(config.optim, config.learning_rate_sample, config.max_grad_norm,
                         lr_decay=config.learning_rate_decay, start_decay_at=config.start_decay_at)
    optim_generate = Optim(config.optim, config.learning_rate, config.max_grad_norm,
                           lr_decay=config.learning_rate_decay, start_decay_at=config.start_decay_at)


    optim_sample.set_parameters(model_sample.parameters())
    optim_generate.set_parameters(model_generate.parameters())
    if config.schedule:
        scheduler_sample = L.SetLR(optim_sample.optimizer)
        scheduler_generate = L.SetLR(optim_generate.optimizer)

    else:
        scheduler_sample = None
        scheduler_generate = None


    if args.type=='train':

        start_time = time.time()

        #dataloader_train = get_dataloader(vocab, split='train', train_type='qk')
        dataloader_train = get_dataloader(vocab, split='train', train_type='k')
        dataloader_dev = get_dataloader(vocab, split='dev', train_type='k')
        dataloader_dev_qk = get_dataloader(vocab, split='dev', train_type='qk')

        print('loading data...\n')
        print('loading time cost: %.3f' % (time.time() - start_time))


        max_bleu = train(model_sample, model_generate, vocab, dataloader_train, dataloader_dev,dataloader_dev_qk,scheduler_sample, scheduler_generate,
                         optim_sample, optim_generate,
                         updates)
        logging("Best bleu score: %.2f\n" % (max_bleu))

    elif args.type == 'eval':
        # Load data
        start_time = time.time()

        dataloader_test = get_dataloader(vocab, split='test', train_type='k')
        dataloader_test_qk = get_dataloader(vocab, split='test', train_type='qk')


        print('loading data...\n')
        print('loading time cost: %.3f' % (time.time() - start_time))
        assert args.restore is not None
        eval(model_sample, model_generate, vocab, dataloader_test,dataloader_test_qk, 0, updates)
    else:
        print('error')


if __name__ == '__main__':
    main()
