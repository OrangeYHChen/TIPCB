import torchvision.transforms as transforms
import torch
import yaml
from function import *
from test_config import parse_args
import time
from models.model import Network
import os
import shutil
import torch.backends.cudnn as cudnn
# from tensorboard_logger import configure, log_value

def test(data_loader, network, args):

    # switch to evaluate mode
    network.eval()
    max_size = args.batch_size * len(data_loader)
    images_bank = torch.zeros((max_size, args.feature_size)).cuda()
    text_bank = torch.zeros((max_size, args.feature_size)).cuda()
    labels_bank = torch.zeros(max_size).cuda()
    index = 0
    with torch.no_grad():
        for images, captions, labels, mask in data_loader:
            images = images.cuda()
            captions = captions.cuda()
            mask = mask.cuda()

            interval = images.shape[0]
            image_embeddings, text_embeddings = network(images, captions, mask)

            images_bank[index: index + interval] = image_embeddings
            text_bank[index: index + interval] = text_embeddings
            labels_bank[index:index + interval] = labels

            index = index + interval

        images_bank = images_bank[:index]
        text_bank = text_bank[:index]
        labels_bank = labels_bank[:index]
        # we input the two times of images, so we need to select half of them
        ac_top1_t2i, ac_top5_t2i, ac_top10_t2i, mAP = test_map(text_bank, labels_bank, images_bank[::2], labels_bank[::2])
        return ac_top1_t2i, ac_top5_t2i, ac_top10_t2i, mAP

def main(model, args):
    test_transform = transforms.Compose([
        transforms.Resize((args.height, args.width), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    test_loaders = data_config(args.dir, batch_size=args.batch_size, split='test', max_length=args.max_length,
                              embedding_type=args.embedding_type, transform=test_transform)

    ac_t2i_top1_best = 0.0
    ac_t2i_top5_best = 0.0
    ac_t2i_top10_best = 0.0
    mAP_best = 0.0
    best = 0
    dst_best = args.checkpoint_dir + "/model_best" + ".pth.tar"

    for i in range(0, args.num_epoches):
        i = i+1
        model_file = os.path.join(args.model_path, str(i))+".pth.tar"
        print(model_file)
        # model_file = os.path.join(args.model_path, 'model_best.pth.tar')
        if os.path.isdir(model_file):
            continue
        start, network = load_checkpoint(model, model_file)
        ac_top1_t2i, ac_top5_t2i, ac_top10_t2i, mAP = test(test_loaders, network, args)
        if ac_top1_t2i > ac_t2i_top1_best:
            ac_t2i_top1_best = ac_top1_t2i
            ac_t2i_top5_best = ac_top5_t2i
            ac_t2i_top10_best = ac_top10_t2i
            mAP_best = mAP
            best = i
            shutil.copyfile(model_file, dst_best)

    print('Epoch:{}:t2i_top1_best: {:.5f}, t2i_top5_best: {:.5f},t2i_top10_best: {:.5f},'
          'mAP_best: {:.5f}'.format(
            best, ac_t2i_top1_best, ac_t2i_top5_best, ac_t2i_top10_best, mAP_best))

def start_test():
    args = parse_args()

    sys.stdout = Logger(os.path.join(args.log_test_dir, "test_log.txt"))

    # load GPU
    str_ids = args.gpus.split(',')
    gpu_ids = []
    for str_id in str_ids:
        gid = int(str_id)
        if gid >= 0:
            gpu_ids.append(gid)

    # set gpu ids
    if len(gpu_ids) > 0:
        torch.cuda.set_device(gpu_ids[0])
        cudnn.benchmark = True
    with open('%s/opts_test.yaml' % args.log_test_dir, 'w') as fp:
        yaml.dump(vars(args), fp, default_flow_style=False)
    model = Network(args).cuda()
    main(model, args)


if __name__ == '__main__':

    args = parse_args()

    sys.stdout = Logger(os.path.join(args.log_test_dir, "test_log.txt"))

    # load GPU
    str_ids = args.gpus.split(',')
    gpu_ids = []
    for str_id in str_ids:
        gid = int(str_id)
        if gid >= 0:
            gpu_ids.append(gid)

    # set gpu ids
    if len(gpu_ids) > 0:
        torch.cuda.set_device(gpu_ids[0])
        cudnn.benchmark = True
    with open('%s/opts_test.yaml' % args.log_test_dir, 'w') as fp:
        yaml.dump(vars(args), fp, default_flow_style=False)
    model = Network(args).cuda()
    main(model, args)
