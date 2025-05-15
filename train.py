import os
import time
import shutil

import torch
import numpy

import data
import opts
from vocab import Vocabulary, deserialize_vocab
from model import SGRAF
from evaluation import i2t, t2i, AverageMeter, LogCollector, encode_data, shard_attn_scores

import logging
import tensorboard_logger as tb_logger


os.environ["CUDA_VISIBLE_DEVICES"] = "3"
def logging_func(log_file, message):
    with open(log_file,'a') as f:
        f.write(message)
    f.close()

def main():
    opt = opts.parse_opt()
    if opt.module_name == 'SGR':
       opt.logger_name='./runG/runX/log'
       opt.logg_path='./runG/runX/logs'
       opt.model_name='./runG/runX/checkpoint'
    elif opt.module_name == 'SAF':
        opt.logger_name = './runA/runX/log'
        opt.logg_path = './runA/runX/logs'
        opt.model_name = './runA/runX/checkpoint'


    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    tb_logger.configure(opt.logger_name, flush_secs=5)
    logger = logging.getLogger(__name__)
    logger.info(opt)
    # Load Vocabulary Wrapper
    vocab = deserialize_vocab(os.path.join(opt.vocab_path, '%s_vocab.json' % opt.data_name))
    opt.vocab_size = len(vocab)

    # Load data loaders
    train_loader, val_loader = data.get_loaders(opt.data_name, vocab, opt.batch_size, opt.workers, opt)

    # Construct the model
    model = SGRAF(opt)
    start_epoch = 0
    if opt.resume:
        if os.path.isfile(opt.resume):

            logger.info("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            start_epoch = checkpoint['epoch']
            best_rsum = checkpoint['best_rsum']
            model.load_state_dict(checkpoint['model'])
            # Eiters is used to show logs as the continuation of another
            # training
            model.Eiters = checkpoint['Eiters']
            logger.info("=> loaded checkpoint '{}' (epoch {}, best_rsum {})"
                        .format(opt.resume, start_epoch, best_rsum))
            # wether validate the resume model
            # validate(opt, val_loader, model)


        else:
            logger.info("=> no checkpoint found at '{}'".format(opt.resume))
    # Train the Model
    best_rsum = 0

    for epoch in range(start_epoch,opt.num_epochs):
        logger.info(opt.logger_name)
        logger.info(opt.model_name)

        adjust_learning_rate(opt, model.optimizer, epoch)

        # train for one epoch
        train(opt, train_loader, model, epoch, val_loader)

        # evaluate on validation set
        r_sum , r1, r5, r10, r1i, r5i, r10i= validate(opt, val_loader, model,epoch)

        # remember best R@ sum and save checkpoint
        is_best = r_sum > best_rsum
        best_rsum = max(r_sum, best_rsum)

        if not os.path.exists(opt.model_name):
            os.mkdir(opt.model_name)
        save_checkpoint({
            'epoch': epoch + 1,
            'model': model.state_dict(),
            'best_rsum': best_rsum,
            'opt': opt,
            'Eiters': model.Eiters,
        }, is_best, filename='checkpoint_{}.pth.tar'.format(epoch), prefix=opt.model_name + '/')


def train(opt, train_loader, model, epoch, val_loader):
    # average meters to record the training statistics
    batch_time = AverageMeter()
    data_time = AverageMeter()
    train_logger = LogCollector()

    end = time.time()
    for i, train_data in enumerate(train_loader):
        # switch to train mode
        model.train_start()

        # measure data loading time
        data_time.update(time.time() - end)

        # make sure train logger is used
        model.logger = train_logger

        # Update the model
        model.train_emb(*train_data)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # Print log info
        if model.Eiters % opt.log_step == 0:
            logging.info(
                'Epoch: [{0}][{1}/{2}]\t'
                '{e_log}\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                .format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, e_log=str(model.logger)))



def validate(opt, val_loader, model,epoch):
    # compute the encoding for all the validation images and captions
    img_embs, cap_embs, cap_lens, adjs = encode_data(model, val_loader, opt.log_step, logging.info)

    # clear duplicate 5*images and keep 1*images
    img_embs = numpy.array([img_embs[i] for i in range(0, len(img_embs), 5)])
    adjs = numpy.array([adjs[i] for i in range(0, len(adjs), 5)])#添加

    # record computation time of validation
    start = time.time()
    sims = shard_attn_scores(model, img_embs, cap_embs, cap_lens, adjs, opt, shard_size=100)
    end = time.time()
    print("calculate similarity time:", end-start)

    # caption retrieval
    (r1, r5, r10, medr, meanr) = i2t(img_embs, cap_embs, cap_lens, sims)
    logging.info("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" % (r1, r5, r10, medr, meanr))

    # image retrieval
    (r1i, r5i, r10i, medri, meanr) = t2i(img_embs, cap_embs, cap_lens, sims)
    logging.info("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" % (r1i, r5i, r10i, medri, meanr))

    # sum of recalls to be used for early stopping
    r_sum = r1 + r5 + r10 + r1i + r5i + r10i
    message = "Epoch: %d:Image to text: (%.1f, %.1f, %.1f) " % (epoch, r1, r5, r10)
    message += "Text to image: (%.1f, %.1f, %.1f) " % (r1i, r5i, r10i)
    message += "rsum: %.1f\n" % r_sum
    log_file = os.path.join(opt.logg_path, "performance.log")
    logging_func(log_file, message)


    return r_sum, r1, r5, r10, r1i, r5i, r10i
def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', prefix=''):
    logger = logging.getLogger(__name__)
    tries = 15
    error = None

    # deal with unstable I/O. Usually not necessary.
    while tries:
        try:
            torch.save(state, prefix + filename)
            if is_best:
                shutil.copyfile(prefix + filename, prefix + 'model_best.pth.tar')
        except IOError as e:
            error = e
            tries -= 1
        else:
            break
        logger.info('model save {} failed, remaining {} trials'.format(filename, tries))
        if not tries:
            raise error
def adjust_learning_rate(opt, optimizer, epoch):
    """
    Sets the learning rate to the initial LR
    decayed by 10 after opt.lr_update epoch
    """
    lr = opt.learning_rate * (0.1 ** (epoch // opt.lr_update))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    main()
