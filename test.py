# -*- coding:utf-8 -*-
import os
import time
import string
import argparse
import re
import logging
import json
import fire
import os
import lmdb
import cv2
import shutil

import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F
import numpy as np
from nltk.metrics.distance import edit_distance
from datetime import datetime

from utils import CTCLabelConverter, AttnLabelConverter, Averager
from dataset import hierarchical_dataset, AlignCollate, Batch_Balanced_Dataset
from model import Model


lmdb_path = './input/lmdb/Test/data.mdb'
print("Reading images...")
# ======================================================================================================
data_path = './input/Test'
directory = './input/Test'
if not os.path.exists(directory):
    os.makedirs(directory)

folder_list = []
"""
    with open('./input/gt_Test.txt', 'a') as gt:
        for folder in os.listdir(data_path):
            img_folder = folder + '.jpg'
            folder_list.append(folder)
            img_path = os.path.join(data_path,folder,'images',img_folder)
            label_path = os.path.join(data_path,folder,'data.json')
            target_path =os.path.join(directory,img_folder)
            shutil.copy(img_path,target_path)
            with open(label_path,'r',encoding='UTF-8') as json_file:
                label = json.load(json_file)
                label = label['value']
                label = label.replace('\n','')
                label = label.replace(' ','')

            #fill = 'Test' + '/'+folder+'/'+'images'+'/'+folder +'.jpg'+'\t'+label +'\n'
            fill = 'Copy' + '/'+'Test'+'/' + img_folder +'\t'+label +'\n'
            gt.write(fill)
"""
gt_file_path = './input/gt_Test.txt'
if not os.path.isfile(gt_file_path):
    with open('./input/gt_Test.txt', 'a') as gt:
        for folder in os.listdir(data_path):
            label = folder.split('_')[0]
            img_folder = folder
            fill = 'Test' + '/' + img_folder + '\t' + label + '\n'
            gt.write(fill)


def checkImageIsValid(imageBin):
        if imageBin is None:
            return False
        imageBuf = np.frombuffer(imageBin, dtype=np.uint8)
        img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
        imgH, imgW = img.shape[0], img.shape[1]
        if imgH * imgW == 0:
            return False
        return True


def writeCache(env, cache):
        with env.begin(write=True) as txn:
            for k, v in cache.items():
                txn.put(k, v)


def createDataset(checkValid=True):
        """
        Create LMDB dataset for training and evaluation.
        ARGS:
            inputPath  : input folder path where starts imagePath
            outputPath : LMDB output path
            gtFile     : list of image path and label
            checkValid : if true, check the validity of every image
        """
        outputPath = './input/lmdb/Test'
        os.makedirs(outputPath, exist_ok=True)
        env = lmdb.open(outputPath, map_size=1099511627)
        cache = {}
        cnt = 1

        gtFile = './input/gt_Test.txt'
        with open(gtFile, 'r', encoding='UTF8') as data:
            datalist = data.readlines()

        nSamples = len(datalist)
        for i in range(nSamples):
            imagePath, label = datalist[i].strip('\n').split('\t')

            # print('imagepath',imagePath)
            inputPath = './input'
            imagePath = os.path.join(inputPath, imagePath)
            # print('imagepath2',imagePath)

            # # only use alphanumeric data
            # if re.search('[^a-zA-Z0-9]', label):
            #     continue

            if not os.path.exists(imagePath):
                # print('%s does not exist' % imagePath)
                print("Test images not loaded")
                continue
            with open(imagePath, 'rb') as f:
                imageBin = f.read()
            if checkValid:
                try:
                    if not checkImageIsValid(imageBin):
                        print('%s is not a valid image' % imagePath)
                        continue
                except:
                    print('error occured', i)
                    with open(outputPath + '/error_image_log.txt', 'a') as log:
                        log.write('%s-th image data occured error\n' % str(i))
                    continue

            imageKey = 'image-%09d'.encode() % cnt
            labelKey = 'label-%09d'.encode() % cnt
            cache[imageKey] = imageBin
            cache[labelKey] = label.encode()

            if cnt % 1000 == 0:
                writeCache(env, cache)
                cache = {}
                # print('Written %d / %d' % (cnt, nSamples))
            cnt += 1
        nSamples = cnt - 1
        cache['num-samples'.encode()] = str(nSamples).encode()
        writeCache(env, cache)

        #print('Created test dataset with %d samples' % nSamples)
        print("%d samples loaded"%nSamples)
        #os.remove('./input/gt_train.txt')

if __name__ == '__main__':
            fire.Fire(createDataset)
        # ======================================================================================================



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

logging.basicConfig(
    filename='Test_Environment_Log.log',
    filemode='w',
    format='%(asctime)s.%(msecs)03d %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

logging.info('python test.py')
def output_command(command):
    bashCommand = command
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    return output, error

def benchmark_all_eval(model, criterion, converter, opt, calculate_infer_time=False):
    """ evaluation with 10 benchmark evaluation datasets """
    # The evaluation datasets, dataset order is same with Table 1 in our paper.
    eval_data_list = ['IIIT5k_3000', 'SVT', 'IC03_860', 'IC03_867', 'IC13_857',
                      'IC13_1015', 'IC15_1811', 'IC15_2077', 'SVTP', 'CUTE80']

    # # To easily compute the total accuracy of our paper.
    # eval_data_list = ['IIIT5k_3000', 'SVT', 'IC03_867',
    #                   'IC13_1015', 'IC15_2077', 'SVTP', 'CUTE80']

    if calculate_infer_time:
        evaluation_batch_size = 1  # batch_size should be 1 to calculate the GPU inference time per image.
    else:
        evaluation_batch_size = opt.batch_size

    list_accuracy = []
    total_forward_time = 0
    total_evaluation_data_number = 0
    total_correct_number = 0
    #log = open(f'./result/{opt.exp_name}/log_all_evaluation.txt', 'a',encoding='UTF8')
    dashed_line = '-' * 80
    print(dashed_line)
    #log.write(dashed_line + '\n')
    for eval_data in eval_data_list:
        eval_data_path = os.path.join(opt.eval_data, eval_data)
        AlignCollate_evaluation = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
        eval_data, eval_data_log = hierarchical_dataset(root=eval_data_path, opt=opt)
        evaluation_loader = torch.utils.data.DataLoader(
            eval_data, batch_size=evaluation_batch_size,
            shuffle=False,
            num_workers=int(opt.workers),
            collate_fn=AlignCollate_evaluation, pin_memory=True)

        _, accuracy_by_best_model, norm_ED_by_best_model, _, _, _, infer_time, length_of_data = validation(
            model, criterion, evaluation_loader, converter, opt)
        list_accuracy.append(f'{accuracy_by_best_model:0.3f}')
        total_forward_time += infer_time
        total_evaluation_data_number += len(eval_data)
        total_correct_number += accuracy_by_best_model * length_of_data
        #log.write(eval_data_log)
        print(f'Acc {accuracy_by_best_model:0.3f}\t normalized_ED {norm_ED_by_best_model:0.3f}')
        #log.write(f'Acc {accuracy_by_best_model:0.3f}\t normalized_ED {norm_ED_by_best_model:0.3f}\n')
        print(dashed_line)
        #log.write(dashed_line + '\n')

    averaged_forward_time = total_forward_time / total_evaluation_data_number * 1000
    total_accuracy = total_correct_number / total_evaluation_data_number
    params_num = sum([np.prod(p.size()) for p in model.parameters()])

    evaluation_log = 'accuracy: '
    for name, accuracy in zip(eval_data_list, list_accuracy):
        evaluation_log += f'{name}: {accuracy}\t'
    evaluation_log += f'total_accuracy: {total_accuracy:0.3f}\t'
    evaluation_log += f'averaged_infer_time: {averaged_forward_time:0.3f}\t# parameters: {params_num / 1e6:0.3f}'
    print(evaluation_log)
    #log.write(evaluation_log + '\n')
    #log.close()

    return None


def validation(model, criterion, evaluation_loader, converter, opt):
    """ validation or evaluation """
    n_total = 0
    n_correct = 0
    n_wrong = 0
    norm_ED = 0
    length_of_data = 0
    infer_time = 0
    valid_loss_avg = Averager()
    Intro = 'Ground Truth , Prediction , Correct/Incorrect'
    logging.info(Intro)
    for i, (image_tensors, labels) in enumerate(evaluation_loader):
        batch_size = image_tensors.size(0)
        length_of_data = length_of_data + batch_size
        image = image_tensors.to(device)
        #print("Index : ",i,"   Length of data :",length_of_data, "   batch_size :",batch_size)
        # For max length prediction
        length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)
        text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)

        text_for_loss, length_for_loss = converter.encode(labels, batch_max_length=opt.batch_max_length)

        start_time = time.time()
        if 'CTC' in opt.Prediction:
            preds = model(image, text_for_pred)
            forward_time = time.time() - start_time

            # Calculate evaluation loss for CTC deocder.
            preds_size = torch.IntTensor([preds.size(1)] * batch_size)
            # permute 'preds' to use CTCloss format
            if opt.baiduCTC:
                cost = criterion(preds.permute(1, 0, 2), text_for_loss, preds_size, length_for_loss) / batch_size
            else:
                cost = criterion(preds.log_softmax(2).permute(1, 0, 2), text_for_loss, preds_size, length_for_loss)

            # Select max probabilty (greedy decoding) then decode index to character
            if opt.baiduCTC:
                _, preds_index = preds.max(2)
                preds_index = preds_index.view(-1)
            else:
                _, preds_index = preds.max(2)
            preds_str = converter.decode(preds_index.data, preds_size.data)

        else:
            preds = model(image, text_for_pred, is_train=False)
            forward_time = time.time() - start_time

            preds = preds[:, :text_for_loss.shape[1] - 1, :]
            target = text_for_loss[:, 1:]  # without [GO] Symbol
            cost = criterion(preds.contiguous().view(-1, preds.shape[-1]), target.contiguous().view(-1))

            # select max probabilty (greedy decoding) then decode index to character
            _, preds_index = preds.max(2)
            preds_str = converter.decode(preds_index, length_for_pred)
            labels = converter.decode(text_for_loss[:, 1:], length_for_loss)

        infer_time += forward_time
        valid_loss_avg.add(cost)

        # calculate accuracy & confidence score
        preds_prob = F.softmax(preds, dim=2)
        preds_max_prob, _ = preds_prob.max(dim=2)
        confidence_score_list = []
        #with open(f'./Failure_log_test.txt', 'a') as log:
        for gt, pred, pred_max_prob in zip(labels, preds_str, preds_max_prob):
                now = time.localtime()
                if 'Attn' in opt.Prediction:
                    gt = gt[:gt.find('[s]')]
                    pred_EOS = pred.find('[s]')
                    pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                    pred_max_prob = pred_max_prob[:pred_EOS]

            # To evaluate 'case sensitive model' with alphanumeric and case insensitve setting.
                if opt.sensitive and opt.data_filtering_off:
                    pred = pred.lower()
                    gt = gt.lower()
                    alphanumeric_case_insensitve = '0123456789abcdefghijklmnopqrstuvwxyz'
                    out_of_alphanumeric_case_insensitve = f'[^{alphanumeric_case_insensitve}]'
                    pred = re.sub(out_of_alphanumeric_case_insensitve, '', pred)
                    gt = re.sub(out_of_alphanumeric_case_insensitve, '', gt)

                if pred == gt:
                    n_correct += 1
                    n_total += 1
                    line=gt,pred,'Correct'
                    logging.info(line)
                    #logging.info(gt,pred,'Correct')
                    #logging.info(str(gt),'|',pred,'|','Correct','|',"%04d/%02d/%02d %02d:%02d:%02d"%(now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec))
                    #logging.info(gt,'|',pred,'|','Correct')


                if pred != gt:
                    n_wrong += 1
                    n_total += 1
                    line = gt,pred,'Incorrect'
                    logging.info(line)
                    #logging.info(gt,pred,'Incorrect')
                    #logging.info(str(gt),'|',pred,'|','Incorrect','|',"%04d/%02d/%02d %02d:%02d:%02d"%(now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec))
                    #logging.info(gt,'|',pred,'|','Incorrect')

                # ICDAR2019 Normalized Edit Distance
                if len(gt) == 0 or len(pred) == 0:
                    norm_ED += 0
                elif len(gt) > len(pred):
                    norm_ED += 1 - edit_distance(pred, gt) / len(gt)
                else:
                    norm_ED += 1 - edit_distance(pred, gt) / len(pred)

            # calculate confidence score (= multiply of pred_max_prob)
                try:
                    confidence_score = pred_max_prob.cumprod(dim=0)[-1]
                except:
                    confidence_score = 0  # for empty pred case, when prune after "end of sentence" token ([s])
                confidence_score_list.append(confidence_score)
            #print(pred, gt, pred==gt, confidence_score)

        logging.info('Correct Prediction : %d'%n_correct)
        logging.info('Incorrect Prediction : %d'%n_wrong)
        logging.info('Total : %d' % n_total)
        accuracy = n_correct / float(length_of_data) * 100
        norm_ED = norm_ED / float(length_of_data)  # ICDAR2019 Normalized Edit Distance
        #log.close
        #print("Iteration : ", i, "  Failure : ", n_wrong)

    return valid_loss_avg.val(), accuracy, norm_ED, preds_str, confidence_score_list, labels, infer_time, length_of_data


def test(opt):
    """ model configuration """
    if 'CTC' in opt.Prediction:
        converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)

    if opt.rgb:
        opt.input_channel = 3
    model = Model(opt)
    #print('model input parameters', opt.imgH, opt.imgW, opt.num_fiducial, opt.input_channel, opt.output_channel,
    #      opt.hidden_size, opt.num_class, opt.batch_max_length, opt.Transformation, opt.FeatureExtraction,
    #      opt.SequenceModeling, opt.Prediction)
    model = torch.nn.DataParallel(model).to(device)

    # load model
    #print('loading pretrained model from %s' % opt.saved_model)
    model.load_state_dict(torch.load(opt.saved_model, map_location=device))
    opt.exp_name = '_'.join(opt.saved_model.split('/')[1:])
    print('Pretrained model loaded')

    # print(model)

    """ keep evaluation model and result logs """
    #os.makedirs(f'./result/{opt.exp_name}', exist_ok=True)
    #os.system(f'cp {opt.saved_model} ./result/{opt.exp_name}/')

    """ setup loss """
    if 'CTC' in opt.Prediction:
        criterion = torch.nn.CTCLoss(zero_infinity=True).to(device)
    else:
        criterion = torch.nn.CrossEntropyLoss(ignore_index=0).to(device)  # ignore [GO] token = ignore index 0

    """ evaluation """
    model.eval()
    with torch.no_grad():
        if opt.benchmark_all_eval:  # evaluation with 10 benchmark evaluation datasets
            benchmark_all_eval(model, criterion, converter, opt)
        else:
            #log = open(f'./result/{opt.exp_name}/log_evaluation.txt', 'a',encoding='UTF8')
            AlignCollate_evaluation = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
            eval_data, eval_data_log = hierarchical_dataset(root=opt.eval_data, opt=opt)
            evaluation_loader = torch.utils.data.DataLoader(
                eval_data, batch_size=opt.batch_size,
                shuffle=False,
                num_workers=int(opt.workers),
                collate_fn=AlignCollate_evaluation, pin_memory=True)
            _, accuracy_by_best_model, _, preds, confidence_score, labels, _, _ = validation(model, criterion, evaluation_loader, converter, opt)

            # show some predicted results
            for gt, pred, confidence in zip(labels[:], preds[:], confidence_score[:]):
                    gt = gt[:gt.find('[s]')]
                    pred = pred[:pred.find('[s]')]

            #log.write(eval_data_log)
            print(f'Accuracy : {accuracy_by_best_model:0.3f}%')
            logging.info(f'Accuracy : {accuracy_by_best_model:0.3f}%')
            #log.write(f'{accuracy_by_best_model:0.3f}\n')
            #log.close()



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_data', default = './input/lmdb/Test', help='path to evaluation dataset')
    parser.add_argument('--benchmark_all_eval', action='store_true', help='evaluate 10 benchmark evaluation datasets')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
    parser.add_argument('--batch_size', type=int, default=192, help='input batch size')
    parser.add_argument('--saved_model', default = './pretrained/Fine-Tuned.pth', help="path to saved_model to evaluation")
    """ Data processing """
    parser.add_argument('--batch_max_length', type=int, default=25, help='maximum-label-length')
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=100, help='the width of the input image')
    parser.add_argument('--rgb', action='store_true', help='use rgb input')
    parser.add_argument('--character', type=str, 
                        #default='0123456789가나다라마거너더러머버서어저고노도로모보소오조구누두루무부수우주하허호배공해육합국서울인천대전대구울산부산광주세종제주강원충북충남전북전남경북경남경기', help='character label')
                        default='0123456789().JNRW_abcdef가강개걍거겅겨견결경계고과관광굥구금기김깅나남너노논누니다대댜더뎡도동두등디라러로루룰리마머명모무문므미바배뱌버베보부북비사산서성세셔소송수시아악안양어여연영오올용우울원육으을이익인자작저전제조종주중지차처천초추출충층카콜타파평포하허호홀후히ㅣ', help='character label')
                        #default='0123456789가강거경계고관광구금기김나남너노누다대더도동두등라러로루마머명모무문미바배버보부북사산서소수아악안양어연영오용우울원육이인자작저전조주중차천초추충카타파평포하허호홀히', help='character label')
                        #default='0123456789가강거경계고관광구금기김나남너노누다대더도동두등라러로루마머명모무문미바배버보부북사산서소송수아악안양어연영오용우울원육이인자작저전제조종주중차천초추충카타파평포하허호홀', help='character label')
    parser.add_argument('--sensitive', action='store_true', help='for sensitive character mode')
    parser.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize')
    parser.add_argument('--data_filtering_off', action='store_true', help='for data_filtering_off mode')
    parser.add_argument('--baiduCTC', action='store_true', help='for data_filtering_off mode')
    """ Model Architecture """
    parser.add_argument('--Transformation', type=str,default='TPS' , help='Transformation stage. None|TPS')
    parser.add_argument('--FeatureExtraction', type=str, default = 'ResNet', help='FeatureExtraction stage. VGG|RCNN|ResNet')
    parser.add_argument('--SequenceModeling', type=str,default = 'BiLSTM' , help='SequenceModeling stage. None|BiLSTM')
    parser.add_argument('--Prediction', type=str, default = 'Attn', help='Prediction stage. CTC|Attn')
    parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
    parser.add_argument('--input_channel', type=int, default=1, help='the number of input channel of Feature extractor')
    parser.add_argument('--output_channel', type=int, default=512,
                        help='the number of output channel of Feature extractor')
    parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')

    opt = parser.parse_args()

    """ vocab / character number configuration """
    if opt.sensitive:
        opt.character = string.printable[:-6]  # same with ASTER setting (use 94 char).

    cudnn.benchmark = True
    cudnn.deterministic = True
    opt.num_gpu = torch.cuda.device_count()

    test(opt)
