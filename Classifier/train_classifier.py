import ujson as json
import numpy as np
from tqdm import tqdm
import os
from torch import optim, nn
from model import Model #, NoCharModel, NoSelfModel
from sp_model import SPModel
from classifier import Classifier
# from normal_model import NormalModel, NoSelfModel, NoCharModel, NoSentModel
# from oracle_model import OracleModel, OracleModelV2
# from util import get_record_parser, convert_tokens, evaluate, get_batch_dataset, get_dataset
from util import convert_tokens, evaluate,  evaluate_classifier
from util import get_buckets, DataIterator, IGNORE_INDEX
import time
import shutil
import random
import torch
from torch.autograd import Variable
import sys
from torch.nn import functional as F
from tensorboardX import SummaryWriter
import time
# from allennlp.modules.elmo import Elmo, batch_to_ids
#
# options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
# weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

# python main.py --mode train --para_limit 2250 --batch_size 24 --init_lr 0.1 --keep_prob 1.0 --sp_lambda 1.0 --total_num_of_buckets 10

def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.mkdir(path)

    print('Experiment dir : {}'.format(path))
    if scripts_to_save is not None:
        if not os.path.exists(os.path.join(path, 'scripts')):
            os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)
weights = torch.cuda.FloatTensor([1, 11]).view(-1)
nll_sum = nn.CrossEntropyLoss(size_average=False, ignore_index=IGNORE_INDEX)
nll_average = nn.CrossEntropyLoss(size_average=True, ignore_index=IGNORE_INDEX)
nll_all = nn.CrossEntropyLoss(weight = weights, reduce=False, ignore_index=IGNORE_INDEX)
classifier_loss = nn.CrossEntropyLoss(weight = weights, reduce=True)

# hardcode the where the best model is saved
best_model_path = 'best_model' + str(time.time()) + '.pth'

def train_classifier(config):
    with open(config.word_emb_file, "r") as fh:
        word_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.char_emb_file, "r") as fh:
        char_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.dev_eval_file, "r") as fh:
        dev_eval_file = json.load(fh)
    with open(config.idx2word_file, 'r') as fh:
        idx2word_dict = json.load(fh)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    # torch.cpu.manual_seed_all(config.seed)

    config.save = '{}-{}'.format(config.save, time.strftime("%Y%m%d-%H%M%S"))
    tbx = SummaryWriter(config.save)
    create_exp_dir(config.save, scripts_to_save=['run.py', 'model.py', 'util.py', 'sp_model.py'])
    def logging(s, print_=True, log_=True):
        if print_:
            print(s)
        if log_:
            with open(os.path.join(config.save, 'log.txt'), 'a+') as f_log:
                f_log.write(s + '\n')

    logging('Config')
    for k, v in config.__dict__.items():
        logging('    - {} : {}'.format(k, v))

    logging("Building model...")
    # train_buckets = get_buckets(config.train_record_file)
    dev_buckets = get_buckets(config.dev_record_file)

    def build_train_iterator(num_of_bucket):
        train_record_file_path = config.train_record_file[:-4] + str(num_of_bucket) + '.pkl'
        train_buckets = get_buckets(train_record_file_path)
        return DataIterator(train_buckets, config.batch_size, config.para_limit, config.ques_limit, config.char_limit, True, config.sent_limit)

    def build_dev_iterator():
        return DataIterator(dev_buckets, config.batch_size, config.para_limit, config.ques_limit, config.char_limit, False, config.sent_limit)

    if config.sp_lambda > 0:
        model = Classifier(config, word_mat, char_mat)
    else:
        model = Model(config, word_mat, char_mat)

    logging('nparams {}'.format(sum([p.nelement() for p in model.parameters() if p.requires_grad])))
    ori_model = model.cuda()
    # ori_model = model.cpu()
    model = nn.DataParallel(ori_model)

    lr = config.init_lr
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=config.init_lr)
    cur_patience = 0
    total_loss = 0
    global_step = 0
    best_dev_F1 = None
    stop_train = False
    start_time = time.time()
    eval_start_time = time.time()
    model.train()
    best_accuracy = 0
    total_num_of_buckets = config.total_num_of_buckets
    for epoch in range(8):####
        for num_of_buckets in range(total_num_of_buckets):
            print ("building data iterator of bucket {} / {}".format(num_of_buckets, total_num_of_buckets))
            for data in build_train_iterator(num_of_buckets):
                context_idxs = Variable(data['context_idxs'])
                ques_idxs = Variable(data['ques_idxs'])
                context_char_idxs = Variable(data['context_char_idxs'])
                ques_char_idxs = Variable(data['ques_char_idxs'])
                context_lens = Variable(data['context_lens'])
                y1 = Variable(data['y1'])
                y2 = Variable(data['y2'])
                q_type = Variable(data['q_type'])
                is_support = Variable(data['is_support'])
                start_mapping = Variable(data['start_mapping'])
                end_mapping = Variable(data['end_mapping'])
                all_mapping = Variable(data['all_mapping'])


                model_starting_time = time.time()

                logit1, logit2, predict_type, predict_support = model(context_idxs, ques_idxs, context_char_idxs, ques_char_idxs, context_lens, start_mapping, end_mapping, all_mapping, return_yp=False)
                loss_1 = (nll_sum(predict_type, q_type) + nll_sum(logit1, y1) + nll_sum(logit2, y2)) / context_idxs.size(0)
                # magnified_predict_support = torch.tensor(predict_support * 100)
                # magnified_is_support = torch.tensor(is_support * 100)
                # print('!!!', magnified_predict_support.size(), magnified_is_support.size())
                loss_2 = classifier_loss(predict_support.view(-1, 2), is_support.view(-1)) # 1 is supporting fact
                loss = loss_1 + 2.0 * loss_2
                tbx.add_scalar('loss', loss, global_step)


                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                model_backprop_time = time.time()

                #total_loss += loss.data[0] # for gpu
                total_loss += loss.item() # for cpu

                global_step += 1

                # print('training 1 batch time is ', model_backprop_time - model_starting_time)

                if global_step % config.period == 0:
                    cur_loss = total_loss / config.period
                    elapsed = time.time() - start_time
                    logging('| epoch {:3d} | step {:6d} | lr {:05.5f} | ms/batch {:5.2f} | train loss {:8.3f}'.format(epoch, global_step, lr, elapsed*1000/config.period, cur_loss))
                    total_loss = 0
                    start_time = time.time()

                if global_step % config.checkpoint == 0:
                    model.eval()
                    metrics, cumulative_accuracy = evaluate_batch_classifier(build_dev_iterator(), model, 0, dev_eval_file, config, global_step, tbx)
                    # if the classifier has a higher accuracy then save it
                    if cumulative_accuracy > best_accuracy:
                        torch.save(model.state_dict(), best_model_path)
                    model.train()

                    logging('-' * 89)
                    logging('| eval {:6d} in epoch {:3d} | time: {:5.2f}s | dev loss {:8.3f} | EM {:.4f} | F1 {:.4f}'.format(global_step//config.checkpoint,
                        epoch, time.time()-eval_start_time, metrics['loss'], metrics['exact_match'], metrics['f1']))
                    logging('-' * 89)


                    for k, v in metrics.items():
                        tbx.add_scalar('dev/{}'.format(k), v, global_step)
                    eval_start_time = time.time()

                    dev_F1 = metrics['f1']
                    if best_dev_F1 is None or dev_F1 > best_dev_F1:
                        best_dev_F1 = dev_F1
                        torch.save(ori_model.state_dict(), os.path.join(config.save, 'model.pt'))
                        cur_patience = 0
                    else:
                        cur_patience += 1
                        if cur_patience >= config.patience:
                            lr /= 2.0
                            for param_group in optimizer.param_groups:
                                param_group['lr'] = lr
                            if lr < config.init_lr * 1e-2:
                                stop_train = True
                                print('lr ',lr)
                                break
                            cur_patience = 0

            if stop_train: break
    logging('best_dev_F1 {}'.format(best_dev_F1))

def evaluate_batch_classifier(data_source, model, max_batches, eval_file, config, step=0, tbx=None):
    answer_dict = {}
    sp_dict = {}
    total_loss, step_cnt = 0, 0
    iter = data_source
    wrong = 0
    total = 0
    sp_th = config.sp_threshold
    total_true_positive = 0
    total_true_negative = 0
    total_false_positive = 0
    total_false_negative = 0
    for step, data in enumerate(iter):
        if step >= max_batches and max_batches > 0: break

        context_idxs = Variable(data['context_idxs'], volatile=True)
        ques_idxs = Variable(data['ques_idxs'], volatile=True)
        context_char_idxs = Variable(data['context_char_idxs'], volatile=True)
        ques_char_idxs = Variable(data['ques_char_idxs'], volatile=True)
        context_lens = Variable(data['context_lens'], volatile=True)
        y1 = Variable(data['y1'], volatile=True)
        y2 = Variable(data['y2'], volatile=True)
        q_type = Variable(data['q_type'], volatile=True)
        is_support = Variable(data['is_support'], volatile=True)
        start_mapping = Variable(data['start_mapping'], volatile=True)
        end_mapping = Variable(data['end_mapping'], volatile=True)
        all_mapping = Variable(data['all_mapping'], volatile=True)

        logit1, logit2, predict_type, predict_support, yp1, yp2 = model(context_idxs, ques_idxs, context_char_idxs, ques_char_idxs, context_lens, start_mapping, end_mapping, all_mapping, return_yp=True)
        loss_2 = classifier_loss(predict_support.view(-1, 2), is_support.view(-1))  # 1 is supporting fact
        loss_1 = (nll_sum(predict_type, q_type) + nll_sum(logit1, y1) + nll_sum(logit2, y2)) / context_idxs.size(0)
        # loss_2 = nll_average(predict_support.view(-1, 2), is_support.view(-1))
        loss = loss_1 + 2.0 * loss_2
        sub_ans = predict_support.view(-1, 2)

        # print('The prediction from the classifier is', sub_ans)
        # print('The shape of prediction we get is', sub_ans.size())
        print(sub_ans)
        predict_support_np = torch.sigmoid(sub_ans)
        # print ('predict_support_np', predict_support_np)
        is_sup_ans = (predict_support_np[:, 1] > sp_th).view(-1)

        print ('is sup ans: ', is_sup_ans.size())
        is_support_reshaped = is_support.view(-1)

        # print ('The prection is', sub_ans)
        print ('The prediction-indexes are', is_sup_ans)
        # print ('The labels are', is_support.view(-1))
        # print ('The masked is_support is', is_support_masked)

        true_positive = torch.sum((is_sup_ans > 0).long() * (is_support_reshaped > 0).long()).data.item()
        print ('true positive: ', true_positive)
        true_negative = torch.sum((is_sup_ans <= 0).long() * (is_support_reshaped <= 0).long()).data.item()
        print ('true negative: ', true_negative)
        false_positive = torch.sum((is_sup_ans > 0).long() * (is_support_reshaped <= 0).long()).data.item()
        print ('false positive: ', false_positive)
        false_negative = torch.sum((is_sup_ans <= 0).long() * (is_support_reshaped > 0).long()).data.item()
        print ('false negative: ', false_negative)
        total_true_positive += true_positive
        total_true_negative += true_negative
        total_false_negative += false_negative
        total_false_positive += false_positive
        # precision = true_positive * 1.0 / (true_positive + false_positive)
        # recall = false_positive * 1.0 / (true_positive + false_negative)
        print (89*"=")
        # print('accurate: ', current_total - current_wrong, 'total: ', current_total)
        # print('the accuracy for current example is', 1.0 * (current_total - current_wrong) / current_total)
        print('total true positive: ', total_true_positive, 'total true negative: ', total_true_negative, 'total false nagative: ', total_false_negative, 'total false positive: ', total_false_positive)
        # print('the precision for current example is', precision)
        # print ('the recall for current example is', recall)
        print (89*"=")
        answer_dict_ = convert_tokens(eval_file, data['ids'], yp1.data.cpu().numpy().tolist(), yp2.data.cpu().numpy().tolist(), np.argmax(predict_type.data.cpu().numpy(), 1))
        answer_dict.update(answer_dict_)
        #dictionary of predictions is answer_dict
        total_loss += loss.data.item()
        step_cnt += 1
    loss = total_loss / step_cnt
    # is_sup_accuracy = (total - wrong) / total
    metrics = evaluate_classifier(eval_file, answer_dict, total_true_positive * 1.0 / (total_true_positive + total_false_negative), step, config, tbx, answer_dict, eval_file, 'val', 10)
    metrics['loss'] = loss
    # classifier_accuracy = is_sup_accuracy
    return (metrics, total_true_positive * 1.0 / (total_true_positive + total_false_negative))


# def visualize(tbx, pred_dict, eval_path, step, split, num_visuals):
#     """Visualize text examples to TensorBoard.
#     Args:
#         tbx (tensorboardX.SummaryWriter): Summary writer.
#         pred_dict (dict): dict of predictions of the form id -> pred.
#         eval_path (str): Path to eval JSON file.
#         step (int): Number of examples seen so far during training.
#         split (str): Name of data split being visualized.
#         num_visuals (int): Number of visuals to select at random from preds.
#     """
#     if num_visuals <= 0:
#         return
#     if num_visuals > len(pred_dict):
#         num_visuals = len(pred_dict)
#
#     visual_ids = np.random.choice(list(pred_dict), size=num_visuals, replace=False)
#
#     with open(eval_path, 'r') as eval_file:
#         eval_dict = json.load(eval_file)
#     for i, id_ in enumerate(visual_ids):
#         pred = pred_dict[id_] or 'N/A'
#         example = eval_dict[str(id_)]
#         question = example['question']
#         context = example['context']
#         answers = example['answer']
#
#         gold = answers[0] if answers else 'N/A'
#         tbl_fmt = ('- **Question:** {}\n'
#                    + '- **Context:** {}\n'
#                    + '- **Answer:** {}\n'
#                    + '- **Prediction:** {}')
#         tbx.add_text(tag='{}/{}_of_{}'.format(split, i + 1, num_visuals),
#                      text_string=tbl_fmt.format(question, context, gold, pred),
#                      global_step=step)



def predict_classifier(data_source, model, eval_file, config, prediction_file):
    answer_dict = {}
    sp_dict = {}
    sp_th = config.sp_threshold
    for step, data in enumerate(tqdm(data_source)):
        context_idxs = Variable(data['context_idxs'], volatile=True)
        ques_idxs = Variable(data['ques_idxs'], volatile=True)
        context_char_idxs = Variable(data['context_char_idxs'], volatile=True)
        ques_char_idxs = Variable(data['ques_char_idxs'], volatile=True)
        context_lens = Variable(data['context_lens'], volatile=True)
        start_mapping = Variable(data['start_mapping'], volatile=True)
        end_mapping = Variable(data['end_mapping'], volatile=True)
        all_mapping = Variable(data['all_mapping'], volatile=True)

        logit1, logit2, predict_type, predict_support, yp1, yp2 = model(context_idxs, ques_idxs, context_char_idxs, ques_char_idxs, context_lens, start_mapping, end_mapping, all_mapping, return_yp=True)
        answer_dict_ = convert_tokens(eval_file, data['ids'], yp1.data.cpu().numpy().tolist(), yp2.data.cpu().numpy().tolist(), np.argmax(predict_type.data.cpu().numpy(), 1))
        answer_dict.update(answer_dict_)

        predict_support_np = torch.sigmoid(predict_support[:, :, 1]).data.cpu().numpy()
        for i in range(predict_support_np.shape[0]):
            cur_sp_pred = []
            cur_id = data['ids'][i]
            for j in range(predict_support_np.shape[1]):
                if j >= len(eval_file[cur_id]['sent2title_ids']): break
                if predict_support_np[i, j] > sp_th:
                    cur_sp_pred.append(eval_file[cur_id]['sent2title_ids'][j])
            sp_dict.update({cur_id: cur_sp_pred})

    prediction = {'answer': answer_dict, 'sp': sp_dict}
    with open(prediction_file, 'w') as f:
        json.dump(prediction, f)

def test_classifier(config):
    with open(config.word_emb_file, "r") as fh:
        word_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.char_emb_file, "r") as fh:
        char_mat = np.array(json.load(fh), dtype=np.float32)
    if config.data_split == 'dev':
        with open(config.dev_eval_file, "r") as fh:
            dev_eval_file = json.load(fh)
    else:
        with open(config.test_eval_file, 'r') as fh:
            dev_eval_file = json.load(fh)
    with open(config.idx2word_file, 'r') as fh:
        idx2word_dict = json.load(fh)

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    #torch.manual_seed_all(config.seed)

    def logging(s, print_=True, log_=True):
        if print_:
            print(s)
        if log_:
            with open(os.path.join(config.save, 'log.txt'), 'a+') as f_log:
                f_log.write(s + '\n')

    if config.data_split == 'dev':
        dev_buckets = get_buckets(config.dev_record_file)
        para_limit = config.para_limit
        ques_limit = config.ques_limit
    elif config.data_split == 'test':
        para_limit = None
        ques_limit = None
        dev_buckets = get_buckets(config.test_record_file)

    def build_dev_iterator():
        return DataIterator(dev_buckets, config.batch_size, para_limit,
            ques_limit, config.char_limit, False, config.sent_limit)

    if config.sp_lambda > 0:
        model = Classifier(config, word_mat, char_mat)
    else:
        model = Model(config, word_mat, char_mat)
    ori_model = model.cuda()
    # ori_model = model.cpu()
    ori_model.load_state_dict(torch.load(os.path.join(config.save, 'model.pt')))
    model = nn.DataParallel(ori_model)

    model.eval()
    predict_classifier(build_dev_iterator(), model, dev_eval_file, config, config.prediction_file)

