import os
import csv
import argparse
from glob import glob


parser = argparse.ArgumentParser(description='generate tiered-ImageNet splits')
parser.add_argument('--imagenet_path', help='path to imagenet train folder')

SPLITPATH = os.path.abspath(os.path.dirname(__file__))
SPLITPATH = os.path.join(SPLITPATH, 'tieredImageNet')

def process_img_file(split='train', imgnet_dir='/imagenet/train'):
    assert split in ['train', 'val', 'test', 'trainval']
    split_path =  os.path.join(SPLITPATH, 'raw', '{}.csv'.format(split))
    
    # read classes
    with open(split_path) as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        category_list = []
        class_list = []
        for row in csv_reader:
            category_list.append(row[1])
            class_list.append(row[0])
        category_list = list(set(category_list))
    print(len(category_list), category_list)
    print(len(class_list), class_list)

    # link class to img
    img_list = []
    for i, keys in enumerate(class_list):
        image_path = glob(os.path.join(imgnet_dir, keys, '*'))
        for idx in range(len(image_path)):
            cur_image_path = image_path[idx][len(imgnet_dir)+1:]
            img_list.append((cur_image_path, keys))
        print(i)
    print(len(img_list))

    #write to csv file
    with open(os.path.join(SPLITPATH, '{}.csv'.format(split)), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['img', 'class'])
        for i in range(len(img_list)):
            writer.writerow([img_list[i][0], img_list[i][1]])



if __name__ == '__main__':
    args = parser.parse_args()
    process_img_file('train', imgnet_dir=args.imagenet_path)
    process_img_file('val', imgnet_dir=args.imagenet_path)
    process_img_file('test', imgnet_dir=args.imagenet_path)