import itertools
from typing import List

import hydra
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models
from PIL import Image
from matplotlib import pyplot as plt, patches
from matplotlib.patches import Rectangle
from torch import ops
from torchvision import transforms
from pybx import anchor
from torchvision.ops import RoIPool

'''
    The Region Proposal Network has 

'''

class RegionProposalNet(pl.LightningModule):
    '''
        Ratios = R > 0
        scale = [0, 1]

    '''
    def __init__(self, last_feature_map_size, anc_scales=[0.75, 0.5, 0.25], anc_ratios = [0.5, 1, 1.5]):
        super().__init__()
        self.anc_scale = anc_scales
        self.anc_ratios = anc_ratios
        self.n_anc_boxes = len(anc_scales) * len(anc_ratios)  # number of anchor boxes for each anchor point

        self.fc_offset = nn.Sequential(nn.Conv2d(in_channels=last_feature_map_size, out_channels=1, kernel_size=1),
                                       nn.Flatten(),
                                       nn.LazyLinear(out_features=4),
                                        nn.ReLU())

        self.fc_box_relevance = nn.Sequential(nn.Conv2d(in_channels=last_feature_map_size, out_channels=1, kernel_size=1),
                                       nn.Flatten(),
                                        nn.LazyLinear(out_features=2),
                                        nn.Softmax(dim=1))
    def forward(self, x, image_size):
        '''
            Considering the input x as (Batch_size, Features_map size, H, W)

            1) Create a series of region proposal that would be the same for each image in the batch

        '''
        batch_size = x.shape[0]
        features_maps_size = x.shape[-2:]
        image_sz = features_maps_size
        feature_sz = (5, 5)
        asp_ratio = self.anc_ratios

        bboxes, _ = anchor.bxs(image_sz, feature_sz, asp_ratio) ## Create bboxes
        #print(bboxes)
        bboxes = [torch.Tensor(bb).expand(batch_size, -1) for bb in bboxes]
        n_bboxes = len(bboxes)
        print(len(bboxes))

        '''
            Input: Tensor([BATCH_SIZE, Features map, H, W]) 
            boxes: List of Tensor, each tensor 
        '''
        #roi_pool_obj = RoIPool(output_size=(5, 5),spatial_scale=asp_ratio )

        output = torchvision.ops.roi_pool(x, bboxes, feature_sz)
        K = output.shape[0]
        features_maps_dim = output.shape[1]
        offsets = self.fc_offset(output)
        box_relevance = self.fc_box_relevance(output)
        print(offsets.shape, box_relevance.shape)
        print(output.shape)
        output = output.reshape(batch_size, int(K/batch_size), features_maps_dim, feature_sz[0], feature_sz[1])
        print(output.shape)


'''
    This model extracts basic features from the image,
    Eventually shrinking the size of image itself to handle 
    lower dimensional images

'''
class BackboneNetwork(pl.LightningModule):

    def __init__(self, conv_out_channels: List,
                 kernel_sizes: List):
        super().__init__()
        self.conv_out_channels = conv_out_channels
        self.kernel_sizes = kernel_sizes


        modules = []
        input_channels = 3
        for map_size, k in zip(self.conv_out_channels, self.kernel_sizes):
            modules.append(
                nn.Sequential(nn.Conv2d(in_channels=input_channels,
                                        out_channels=map_size,
                                        kernel_size=k),
                                nn.MaxPool2d(kernel_size=4),
                                nn.BatchNorm2d(map_size)
                              )

            )
            input_channels = map_size

        self.back_bone = nn.Sequential(*modules)


    def forward(self, x):
        x = self.back_bone(x)
        return x

'''
    Faster RCNN
    1) Backbone
    2) RegionProposalNetwork (RPN):
        2.1) Produces Anchorboxes
        2.2) From each boxes produces [Offset of the boxes, box score]
    3) 
    

'''

class FasterRCNN(pl.LightningModule):

    def __init__(self, conv_out_channels: List, kernel_sizes: List, lr=1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = lr
        self.backbone = BackboneNetwork(conv_out_channels, kernel_sizes)
        self.rpn = RegionProposalNet(last_feature_map_size=conv_out_channels[-1])

    def forward(self, x):
        image_size = x.shape[-2:]
        x = self.backbone(x) ##Features maps
        x = self.rpn(x, image_size)
        return x


    def training_step(self, batch, batch_idx):
        pass
    def training_epoch_end(self, outputs):
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def validation_epoch_end(self, outputs):
        pass

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)

'''
    TESTING FUNCTIONS
'''
def plotAnchorPoints(anc_pts_x, anc_pts_y, width_scale_factor, height_scale_factor, img):
    anc_pts_x_proj = anc_pts_x.clone() * width_scale_factor
    anc_pts_y_proj = anc_pts_y.clone() * height_scale_factor
    points = list(itertools.product(anc_pts_x_proj, anc_pts_y_proj))
    x_s = [p[0] for p in points]
    y_s = [p[1] for p in points]
    #np_img = torchvision.transforms.ToPILImage()(img)
    implot = plt.imshow(img)
    plt.plot(x_s, y_s, marker='+', c="r", ls='')

    plt.show()

def plotBBoxes(bboxes, img):

    bboxes = bboxes.squeeze()
    bbx = []
    for i in range(bboxes.shape[0]):
        coordinates = bboxes[i, :].squeeze()
        x = coordinates[0]
        y = coordinates[3]
        w = (coordinates[2] - coordinates[0])
        h = (coordinates[1] - coordinates[3])
        bbx.append((x, y, w, h))

    fig, ax = plt.subplots()

    ax.imshow(img)
    for box in bbx:
        rect = patches.Rectangle((box[0], box[1]), box[2], box[3], linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    plt.show()

def show_bboxes(axes, bboxes, labels=None, colors=None):
    """Show bounding boxes."""

    def make_list(obj, default_values=None):
        if obj is None:
            obj = default_values
        elif not isinstance(obj, (list, tuple)):
            obj = [obj]
        return obj

    labels = make_list(labels)
    colors = make_list(colors, ['b', 'g', 'r', 'm', 'c'])
    for i, bbox in enumerate(bboxes):
        color = colors[i % len(colors)]
        rect = torch.bbox_to_rect(bbox.detach().numpy(), color)
        axes.add_patch(rect)
        if labels and len(labels) > i:
            text_color = 'k' if color == 'w' else 'w'
            axes.text(rect.xy[0], rect.xy[1], labels[i],
                      va='center', ha='center', fontsize=9, color=text_color,
                      bbox=dict(facecolor=color, lw=0))


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg):
    image = Image.open("../../Datasets/Flavia/Leaves/1006.jpg")
    transform = transforms.Compose([transforms.ToTensor()])
    image_tensor = transform(image)
    #image_tensor = image_tensor[None, :, :, :]

    image2 = Image.open("../../Datasets/Flavia/Leaves/1007.jpg")
    image2_tensor = transform(image2)

    batch = torch.stack([image_tensor, image2_tensor])
    print(batch.shape)

    model = FasterRCNN(cfg.conv_out_channels, cfg.kernel_sizes)

    out = model(batch)
    print(out.shape)


def test():
    '''
        Get the image
    '''
    image = Image.open("../../Datasets/Flavia/Leaves/1006.jpg")

    '''
        Make some transformations
    '''
    transform = transforms.Compose([transforms.ToTensor()])
    image_tensor = transform(image)

    _, img_width, img_height = image_tensor.shape

    '''
        Get the features map from the image
    '''
    back_bone = BackboneNetwork()
    image_tensor = image_tensor[None, :, :, :] ## convert in [BATCH_SIZE, NUM CHANNELS, WIDTH, HEIGHT]
    out = back_bone(image_tensor) ## Get features map
    out = out.squeeze() ##delete batch_size
    n_channel, f_map_width, f_map_height = out.shape

    width_scale_factor = img_width // f_map_width ## get the difference of the scale between the feature map and the actual image
    height_scale_factor = img_height // f_map_height
    print(width_scale_factor, height_scale_factor)
    '''
        Get Anchor boxes from the RPN
    '''
    rpn = RegionProposalNet()

    bboxes = rpn.generateAnchorBoxes((f_map_width, f_map_height))
    print("bboxes shape", bboxes.shape)
    bbox_scale = torch.tensor((width_scale_factor, height_scale_factor, width_scale_factor, height_scale_factor))
    plotBBoxes(bboxes[:, 450:456, :]*bbox_scale, img=image)
    fig = plt.imshow(image)
    #show_bboxes(fig.axes, bboxes[250, 250, :, :] * bbox_scale,
      #          ['s=0.75, r=1', 's=0.5, r=1', 's=0.25, r=1', 's=0.75, r=2',
      #           's=0.75, r=0.5'])


if __name__ == "__main__":

    #test()
    main()