import argparse
import sys,os
import imghdr,re
import numpy as np
import pandas as pd
from pandas import DataFrame,Series
import torch,os
import torch.nn as nn
import torch.utils.data as data
from torchvision.transforms import v2
from torchvision.models import resnet34
from PIL import Image
from functools import reduce

def parse_args():
    parse = argparse.ArgumentParser(description='Receive the parameters of model')
    parse.add_argument('-i','--image',type=str,metavar='',required=True,help='The path of image for recognizing')
    parse.add_argument('-c','--cpu',type=int,metavar='',required=False,default=2,help='The Cpu for use')
    args = parse.parse_args()
    return args

def recognize(model,img_path):
    img = Image.open(img_path).convert('L')
    preprocess = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.uint8, scale=True),
            v2.Resize([250,250],antialias=True),
            v2.CenterCrop([224, 224]),
            #v2.Grayscale(num_output_channels=1),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    tensor_img = preprocess(img).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        output = model(tensor_img)
        probabilities = nn.functional.softmax(output,dim=1)
        sorted_probabilities,indices = torch.sort(probabilities,descending=True)
        probabilities = sorted_probabilities[0][:3]
        indices = indices[0][:3]
    return probabilities,indices

def genus(folder,index):
    raw_dic = {0: 'Abies', 1: 'Acacia', 2: 'Aster', 3: 'Berberis', 4: 'Camellia', 5: 'Cassia', 6: 'Castanopsis', 7: 'Citrus', 8: 'Clematis', 9:'Cornus', 10: 'Dendrolobium', 11: 'Elaeocarpus',12: 'Euonymus', 13: 'Fraxinus', 14: 'Ilex', 15: 'Indigofera', 16: 'Iris', 17: 'Ligustrum', 18: 'Lonicera', 19: 'Magnolia', 20: 'Michelia', 21: 'Pedicularis', 22: 'Picea', 23: 'Pinus', 24: 'Populus', 25: 'Prunus', 26: 'Quercus', 27: 'Rhododendron', 28: 'Ribes', 29: 'Rosa', 30: 'Salix', 31: 'Symplocos', 32: 'Syringa', 33: 'Tilia', 34: 'Ulmus', 35: 'Viburnum'}
    array = np.array([])
    for i in index:
        array = np.append(array,raw_dic[i])
    return array

def replace_relu_with_elu(model):
    for name, child in model.named_children():
        if isinstance(child, nn.ReLU):
            setattr(model, name, nn.ELU())
        else:
            replace_relu_with_elu(child)

def valid_save_img(img_path):
    valid_formats = {'jpeg','png','tiff'}
    try:
        with Image.open(img_path) as image:
            file_format = imghdr.what(img_path)
            if file_format not in valid_formats:
                raise ValueError('图片格式错误,请输入jpg,png或tiff')

            elif file_format.lower() == 'jpeg':
                if img_path.endswith('.JPG'):
                    re_img_path = img_path.replace('.JPG','.png')
                else:
                    re_img_path = img_path.replace('.jpg', '.png')

            elif file_format.lower() == '.tiff':
                if img_path.endswith('.TIFF'):
                    re_img_path = img_path.replace('.TIFF','.png')
                else:
                    re_img_path = img_path.replace('.tiff','.png')

            elif file_format.lower() == 'png':
                if img_path.endswith('.PNG'):
                    re_img_path = img_path.replace('.PNG','.png')
                else:
                    re_img_path = img_path

            raw_path = '/root/aipollen/'
            re_img_path = os.path.join(raw_path,'upload_data',re_img_path)
            if not os.path.exists(re_img_path):
                image.save(re_img_path)

    except IOError:
        if os.path.exists(img_path):
            os.remove(img_path)
        raise ValueError('文件无法打开,请检查文件路径和文件是否为有效的图片格式')

    return re_img_path

class Genus_name:
    def __init__(self,raw_dic):
        self.dic = raw_dic

    def add(self,new_dic):
        self.dic.update(new_dic)

    def reset(self):
        self.dic = {}

if __name__ == '__main__':
    args = parse_args()
    img_path = args.image
    cpu_num = args.cpu
    torch.set_num_threads(cpu_num)

    try:
        img_path = valid_save_img(img_path)
    except ValueError as e:
        print(e)
        sys.exit(1)

    genus_folder = r".\datasets"
    resnet = resnet34(weights=None)
    #resnet.conv1 = nn.Conv2d(1,64,kernel_size=(7,7),stride=(2,2),padding=(3,3),bias=False)
    inputs = resnet.fc.in_features
    resnet.fc = nn.Linear(inputs,36,bias=True)
    replace_relu_with_elu(resnet)
    checkpoint = torch.load(r".\model.pth",map_location='cpu',weights_only=True)
    resnet.load_state_dict(checkpoint)

    probabilites,indices = recognize(resnet,img_path)
    probabilites = probabilites.data.numpy()
    indices = indices.data.numpy()
    genus_result = genus(genus_folder,indices)
    family = Genus_name({'Abies': 'Pinaceae', 'Picea': 'Pinaceae', 'Pinus': 'Pinaceae', 'Acacia': 'Fabaceae', 'Cassia': 'Fabaceae',
                            'Dendrolobium': 'Fabaceae', 'Indigofera': 'Fabaceae', 'Aster': 'Asteraceae', 'Berberidaceae': 'Berberidaceae',
                            'Camellia': 'Theaceae', 'Castanopsis': 'Fagaceae', 'Quercus': 'Fagaceae', 'Citrus': 'Rutaceae',
                            'Clematis': 'Ranunculaceae', 'Cornus': 'Cornaceae', 'Elaeocarpus': 'Elaeocarpaceae', 'Euonymus': 'Celastraceae ',
                            'Fraxinus': 'Oleaceae', 'Ligustrum': 'Oleaceae', 'Syringa': 'Oleaceae', 'Ilex': 'Aquifoliaceae', 'Iris': 'Iridaceae',
                            'Lonicera': 'Caprifoliaceae', 'Viburnum': 'Caprifoliaceae', 'Magnolia': 'Magnoliaceae', 'Michelia': 'Magnoliaceae',
                            'Pedicularis': 'Orobanchaceae', 'Populus': 'Salicaceae', 'Salix': 'Salicaceae', 'Ribes': 'Grossulariaceae', 'Prunus':
                                'Rosaceae', 'Rosa': 'Rosaceae', 'Rhododendron': 'Ericaceae', 'Symplocos': 'Symplocaceae', 'Tilia': 'Malvaceae',
                            'Ulmus': 'Ulmaceae'})
    order = Genus_name({'Abies': 'Pinales', 'Picea': 'Pinales', 'Pinus': 'Pinales', 'Acacia': 'Fabales', 'Cassia': 'Fabales',
                        'Dendrolobium': 'Fabales', 'Indigofera': 'Fabales', 'Aster': 'Asterales', 'Berberidaceae': 'Ranunculales',
                        'Camellia': 'Ericales', 'Castanopsis': 'Fagales', 'Quercus': 'Fagales', 'Citrus': 'Sapindales', 'Clematis': 'Ranunculales',
                        'Cornus': 'Cornales', 'Elaeocarpus': 'Oxalidales', 'Euonymus': 'Celastrales', 'Fraxinus': 'Lamiales', 'Ligustrum': 'Lamiales',
                        'Syringa': 'Lamiales', 'Ilex': 'Aquifoliales', 'Iris': 'Asparagales', 'Lonicera': 'Dipsacales', 'Viburnum': 'Dipsacales',
                        'Magnolia': 'Magnoliales', 'Michelia': 'Magnoliales', 'Pedicularis': 'Lamiales', 'Populus': 'Malpighiales', 'Salix': 'Malpighiales',
                        'Ribes': 'Saxifragales', 'Prunus': 'Rosales', 'Rosa': 'Rosales', 'Rhododendron': 'Ericales', 'Symplocos': 'Ericales',
                        'Tilia': 'Malvales', 'Ulmus': 'Rosales'})
    # print(f'probabilities:{probabilites}\n')
    # print(f'genus_name:{genus_result}\n')
    family_result = []
    order_result = []
    for genus in genus_result:
        family_result.append(family.dic[genus])
        order_result.append(order.dic[genus])
    # print(f'family_name:{family_result}\n')
    # print(f'order_name:{order_result}\n')
    data = np.stack((order_result,family_result,genus_result,probabilites),axis=1)
    df = DataFrame(data=data)
    df.columns = ['order','family','genus','probabilities']
    df.index = ['result1','result2','result3']
    pd.set_option('display.unicode.ambiguous_as_wide', True)
    pd.set_option('display.unicode.east_asian_width', True)
    pd.set_option('display.width', 180)

    genus_name = df.iloc[0]['genus']
    probabilites = df.iloc[0]['probabilities']
    result = reduce(lambda x,y:x+y,[genus_name,'_',probabilites])

    with Image.open(img_path) as img:
        saved_img_path = re.sub(r'^([^/]*\/[^/]*\/[^/]*\/[^/]*\/).*', r'\1{}.png'.format(result), img_path)
        print(saved_img_path)
        img.save(saved_img_path)

    if os.path.exists(img_path):
        os.remove(img_path)

    print(df)

