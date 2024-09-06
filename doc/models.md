PyTorch provides a wide range of pre-defined models for various tasks through the `torchvision.models` module. These models are primarily used for image classification, object detection, segmentation, and other computer vision tasks. Here are some of the commonly used models:

### Image Classification Models
1. **ResNet (Residual Networks)**
   - `torchvision.models.resnet18`
   - `torchvision.models.resnet34`
   - `torchvision.models.resnet50`
   - `torchvision.models.resnet101`
   - `torchvision.models.resnet152`

2. **VGG (Visual Geometry Group)**
   - `torchvision.models.vgg11`
   - `torchvision.models.vgg13`
   - `torchvision.models.vgg16`
   - `torchvision.models.vgg19`

3. **DenseNet (Densely Connected Convolutional Networks)**
   - `torchvision.models.densenet121`
   - `torchvision.models.densenet169`
   - `torchvision.models.densenet201`
   - `torchvision.models.densenet161`

4. **Inception (GoogLeNet)**
   - `torchvision.models.inception_v3`

5. **MobileNet**
   - `torchvision.models.mobilenet_v2`
   - `torchvision.models.mobilenet_v3_small`
   - `torchvision.models.mobilenet_v3_large`

6. **SqueezeNet**
   - `torchvision.models.squeezenet1_0`
   - `torchvision.models.squeezenet1_1`

7. **EfficientNet**
   - `torchvision.models.efficientnet_b0`
   - `torchvision.models.efficientnet_b1`
   - `torchvision.models.efficientnet_b2`
   - `torchvision.models.efficientnet_b3`
   - `torchvision.models.efficientnet_b4`
   - `torchvision.models.efficientnet_b5`
   - `torchvision.models.efficientnet_b6`
   - `torchvision.models.efficientnet_b7`

8. **AlexNet**
   - `torchvision.models.alexnet`

9. **ShuffleNet**
   - `torchvision.models.shufflenet_v2_x0_5`
   - `torchvision.models.shufflenet_v2_x1_0`
   - `torchvision.models.shufflenet_v2_x1_5`
   - `torchvision.models.shufflenet_v2_x2_0`

10. **MNASNet**
    - `torchvision.models.mnasnet0_5`
    - `torchvision.models.mnasnet0_75`
    - `torchvision.models.mnasnet1_0`
    - `torchvision.models.mnasnet1_3`

### Object Detection and Segmentation Models
1. **Faster R-CNN**
   - `torchvision.models.detection.fasterrcnn_resnet50_fpn`

2. **Mask R-CNN**
   - `torchvision.models.detection.maskrcnn_resnet50_fpn`

3. **RetinaNet**
   - `torchvision.models.detection.retinanet_resnet50_fpn`

4. **SSD (Single Shot MultiBox Detector)**
   - `torchvision.models.detection.ssd300_vgg16`

5. **Keypoint R-CNN**
   - `torchvision.models.detection.keypointrcnn_resnet50_fpn`

### Semantic Segmentation Models
1. **FCN (Fully Convolutional Network)**
   - `torchvision.models.segmentation.fcn_resnet50`
   - `torchvision.models.segmentation.fcn_resnet101`

2. **DeepLabV3**
   - `torchvision.models.segmentation.deeplabv3_resnet50`
   - `torchvision.models.segmentation.deeplabv3_resnet101`

3. **LRASPP (Lightweight RefineNet)**
   - `torchvision.models.segmentation.lraspp_mobilenet_v3_large`

These models can be loaded with pre-trained weights on standard datasets like ImageNet, which can significantly speed up the development process by leveraging transfer learning. To use any of these models, you can simply import and instantiate them, often specifying whether you want to use pre-trained weights. For example:

```python
import torchvision.models as models

# Load a pre-trained ResNet-50 model
model = models.resnet50(pretrained=True)
```

PyTorch also allows for the customization and fine-tuning of these models to suit specific needs.

PyTorch offers a wide range of pre-defined neural network (NN) models in the `torch.nn` module, suitable for various types of tasks like image classification, object detection, and natural language processing. Below is a list of some commonly used neural network models available in PyTorch:

### Convolutional Neural Networks (CNNs)

1. **LeNet**
   - **Description:** An early CNN model designed for handwritten digit classification (MNIST).
   - **Usage:** `torchvision.models.lenet()`

2. **AlexNet**
   - **Description:** A deep CNN model known for its performance on the ImageNet dataset.
   - **Usage:** `torchvision.models.alexnet(pretrained=True)`

3. **VGG**
   - **Description:** A family of models (VGG11, VGG13, VGG16, VGG19) known for their simplicity and depth.
   - **Usage:** `torchvision.models.vgg16(pretrained=True)`

4. **ResNet**
   - **Description:** A deep CNN model with residual connections (ResNet18, ResNet34, ResNet50, ResNet101, ResNet152).
   - **Usage:** `torchvision.models.resnet50(pretrained=True)`

5. **DenseNet**
   - **Description:** A CNN model with dense connections between layers (DenseNet121, DenseNet169, DenseNet201).
   - **Usage:** `torchvision.models.densenet121(pretrained=True)`

6. **Inception**
   - **Description:** A CNN model with Inception modules for improved performance (InceptionV3).
   - **Usage:** `torchvision.models.inception_v3(pretrained=True)`

7. **EfficientNet**
   - **Description:** A model that scales up network width, depth, and resolution efficiently.
   - **Usage:** `torchvision.models.efficientnet_b0(pretrained=True)`

8. **MobileNet**
   - **Description:** A lightweight CNN model for mobile and embedded devices (MobileNetV2, MobileNetV3).
   - **Usage:** `torchvision.models.mobilenet_v2(pretrained=True)`

### Recurrent Neural Networks (RNNs)

1. **RNN**
   - **Description:** Basic RNN model.
   - **Usage:** `torch.nn.RNN(input_size, hidden_size, num_layers)`

2. **LSTM**
   - **Description:** Long Short-Term Memory network, effective for sequence prediction tasks.
   - **Usage:** `torch.nn.LSTM(input_size, hidden_size, num_layers)`

3. **GRU**
   - **Description:** Gated Recurrent Unit, an alternative to LSTM with fewer parameters.
   - **Usage:** `torch.nn.GRU(input_size, hidden_size, num_layers)`

### Transformer Models

1. **Transformer**
   - **Description:** General transformer architecture used for various tasks.
   - **Usage:** `torch.nn.Transformer()`

2. **BERT (Bidirectional Encoder Representations from Transformers)**
   - **Description:** Pre-trained model for natural language understanding.
   - **Usage:** Available through `transformers` library by Hugging Face: `transformers.BertModel.from_pretrained('bert-base-uncased')`

3. **GPT (Generative Pre-trained Transformer)**
   - **Description:** Pre-trained model for text generation tasks.
   - **Usage:** Available through `transformers` library: `transformers.GPT2Model.from_pretrained('gpt2')`

### Other Models

1. **AutoEncoder**
   - **Description:** Neural network used for unsupervised learning and dimensionality reduction.
   - **Usage:** Custom implementation based on `torch.nn.Module`.

2. **GAN (Generative Adversarial Network)**
   - **Description:** Neural network used for generating data samples.
   - **Usage:** Custom implementation with `torch.nn.Module` for generator and discriminator.

### Example of Using Pre-trained Models

Here’s a simple example of how to use a pre-trained ResNet model for image classification:

```python
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image

# Load pre-trained model
model = models.resnet50(pretrained=True)
model.eval()

# Define a transformation to preprocess the image
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load and preprocess an image
input_image = Image.open("path_to_image.jpg")
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0)  # Create a mini-batch as expected by the model

# Make predictions
with torch.no_grad():
    output = model(input_batch)
    
# Output processing (e.g., get the top-5 predictions)
_, predicted_indices = torch.topk(output, k=5)
print(predicted_indices)
```

These models cover a wide range of tasks, from image classification and object detection to natural language processing and generative tasks. For more specialized or advanced models, you might need to look into third-party libraries or custom implementations.