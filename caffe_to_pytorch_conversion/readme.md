# Conversion Steps

## Requirements
* Installing MMdNN 
```
$ - pip3 install mmdnn
```
* Downgrade numpy to 1.16.x

## Covert to IR format

```
$ - mmtoir -f caffe -d garbnet -n deploy_garbnet.prototxt -w garbnet_fcn.caffemodel -o garbnet_IR
```

Options:
1. -f (source framework)
2. -d (destination)
3. -n (network) caffe stores as <filename>.prototxt
4. -w (weights) caffe stores as <filename>.caffemodel
5. -o (output filename)

## Now we convert to code
```
$ - mmtocode -f pytorch -n garbnet_IR.pb --IRWeightPath garbnet_IR.npy --dstModelPath pytorch_garbnet.py -dw pytorch_garbnet.npy
```

Options:
1. -f (output framework)
2. -n (IR network)
3. --IRWeightPath mmdnn stores the weigths as <filename>.pb
4. --dstModelPath (destination file path)
5. -dw (destination weights)

## Finally, converting to pytorch model
```
$ - mmtomodel -f pytorch -in pytorch_garbnet.py -iw pytorch_garbnet.npy --o pytorch_garbnet.pth
```

Options:
1. -f (output framework)
2. -in (input network) stores as <filename>.py
3. -iw (input weights) stores as <filename>.npy
4. --o (output file) specify filename

## Loading the PyTorch model
```python
import torch
# need filepath to network created from mmdnn converter
# because it has the model class defined
MainModel = imp.load_source('MainModel', "SpotGarbage_GarbNet/pytorch_garbnet.py")

# specify the path to the pytorch model for torch.load method
the_model = torch.load("SpotGarbage_GarbNet/pytorch_garbnet.pth")

#print model architecture
the_model.eval()
```

