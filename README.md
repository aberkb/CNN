# Classification of Colorectal pathology images using CNN
##### Pathology is known as the science of the causes and effects of diseases. In particular, it is a branch of medicine that deals with the examination of samples of body tissue in the laboratory for diagnostic or forensic purposes. Hematoxylin is stained with eosin (H & E) in order to make it more visually meaningful.
##### The convolutional neural network, which is frequently used to classify colorectal histapatology images, was used on dataset which contains equal number of elements with 8 different classes. Tumor, stroma, complex, lympho, debris, mucosa. 

![screenshot from 2019-01-12 19-35-53](https://user-images.githubusercontent.com/33849722/51075780-5e59cd00-16a1-11e9-9339-1d9162b8951d.png)

##### There are 4 convolution and 4 fully connected layers in the system used. The number of filters in the Convolution layers is 128 128 64 32. After the last conv + max pool layer, the entry of flatten with fully connected layer is prepared. The FCs have 256, 128 128 and 32 neurons. The learning phase of the first experiments, on cpu (Intel(R) Core(TM) i5-2310 CPU @ 2.90GHz),took around 5.30 hours. After that (high step per epochs and deeper nets) experiments on tesla k80 gpu (google colab) 2.30 hours.
## Architecture of Network

![screenshot from 2019-01-06 21-43-55](https://user-images.githubusercontent.com/33849722/51075752-0f139c80-16a1-11e9-8168-ce84dabb220f.png)

## Results

![screenshot from 2019-01-07 00-01-46](https://user-images.githubusercontent.com/33849722/51075782-6f0a4300-16a1-11e9-97b8-d2577714d0c5.png)
![cnn](https://user-images.githubusercontent.com/33849722/50567242-9aad3500-0d53-11e9-9b17-4c36d3bbcf8f.png)

### Comparison with svm will be done shortly
