import os
import numpy as np
import matplotlib.pyplot as plt
import caffe
import sys
#%matplotlib inline

#caffe.set_device(0)
#caffe.set_mode_gpu()

caffe.set_mode_cpu()

#caffe_root = './caffe-master/' 
#sys.path.insert(0, caffe_root + 'python')


#plt.rcParams['figure.figsize'] = (10, 10)
#plt.rcParams['image.interpolation'] = 'nearest'
#plt.rcParams['image.cmap'] = 'gray'

#mean_filename='./mean.binaryproto'
#proto_data = open(mean_filename, "rb").read()
#a = caffe.io.caffe_pb2.BlobProto.FromString(proto_data)
#mean  = caffe.io.blobproto_to_array(a)[0]


#age_net_pretrained='./dex_chalearn_iccv2015.caffemodel'
#age_net_model_file='./age.prototxt'

age_net_pretrained='./dex_chalearn_iccv2015.caffemodel'
age_net_model_file='./age.prototxt'

age_net = caffe.Classifier(age_net_model_file, age_net_pretrained,
                       #mean=mean,
                       channel_swap=(2,1,0),
                       raw_scale=255,
                       image_dims=(256, 256))



age2_net_pretrained='./dex_imdb_wiki.caffemodel'
age2_net_model_file='./age2.prototxt'

age2_net = caffe.Classifier(age2_net_model_file, age2_net_pretrained,
                       #mean=mean,
                       channel_swap=(2,1,0),
                       raw_scale=255,
                       image_dims=(256, 256))





gender_net_pretrained='./gender.caffemodel'
gender_net_model_file='./gender.prototxt'
gender_net = caffe.Classifier(gender_net_model_file, gender_net_pretrained,
                       #mean=mean,
                       channel_swap=(2,1,0),
                       raw_scale=255,
                       image_dims=(256, 256))



age_list=['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32','33','34','35','36','37','38','39','40','41','42','43','44','45','46','47','48','49','50','51','52','53','54','55','56','57','58','59','60','61','62','63','64','65','66','67','68','69','70','71','72','73','74','75','76','77','78','79','80','81','82','83','84','85','86','87','88','89','90','91','92','93','94','95','96','97','98','99','100']
gender_list=['Female','Male']


example_image = './example.jpg'
input_image = caffe.io.load_image(example_image)
#_ = plt.imshow(input_image)

prediction = age_net.predict([input_image]) 
print 'predicted age1:', age_list[prediction[0].argmax()]

prediction = age2_net.predict([input_image]) 
print 'predicted age2:', age_list[prediction[0].argmax()]

prediction = gender_net.predict([input_image]) 
print 'predicted gender:', gender_list[prediction[0].argmax()]
