import itertools

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
from torchvision.models import ResNet50_Weights

class RegionProposalNet(pl.LightningModule):
    '''
        Ratios = R > 0
        scale = [0, 1]

    '''
    def __init__(self, anc_scales=[0.75, 0.5, 0.25], anc_ratios = [0.5, 1, 1.5]):
        super().__init__()
        self.anc_scale = anc_scales
        self.anc_ratios = anc_ratios
        self.n_anc_boxes = len(anc_scales) * len(anc_ratios)  # number of anchor boxes for each anchor point

    def forward(self, x):
        features_maps_size = x.shape[-2:]
        anchor_boxes = self.generateAnchorBoxes(features_maps_size)



    '''
        Returns bboxes with shape
        (batch_size, number of anchor boxes, 4)
        the 4 elements are (x,y) coordidate of upperleft corner and 
        (x,y) coordinates in the lowerright corner
    '''
    # @save
    def generateAnchorBoxes(self, input_size):
        sizes = self.anc_scale
        ratios = self.anc_ratios
        """Generate anchor boxes with different shapes centered on each pixel."""
        in_height, in_width = input_size
        device, num_sizes, num_ratios = self.device, len(sizes), len(ratios)
        boxes_per_pixel = (num_sizes + num_ratios - 1)
        size_tensor = torch.tensor(sizes, device=device)
        ratio_tensor = torch.tensor(ratios, device=device)
        # Offsets are required to move the anchor to the center of a pixel. Since
        # a pixel has height=1 and width=1, we choose to offset our centers by 0.5
        offset_h, offset_w = 0.5, 0.5
        steps_h = 1.0 / in_height  # Scaled steps in y axis
        steps_w = 1.0 / in_width  # Scaled steps in x axis

        # Generate all center points for the anchor boxes
        center_h = (torch.arange(in_height, device=device) + offset_h) * steps_h
        center_w = (torch.arange(in_width, device=device) + offset_w) * steps_w
        shift_y, shift_x = torch.meshgrid(center_h, center_w, indexing='ij')
        shift_y, shift_x = shift_y.reshape(-1), shift_x.reshape(-1)

        # Generate `boxes_per_pixel` number of heights and widths that are later
        # used to create anchor box corner coordinates (xmin, xmax, ymin, ymax)
        w = torch.cat((size_tensor * torch.sqrt(ratio_tensor[0]),
                       sizes[0] * torch.sqrt(ratio_tensor[1:]))) \
            * in_height / in_width  # Handle rectangular inputs
        h = torch.cat((size_tensor / torch.sqrt(ratio_tensor[0]),
                       sizes[0] / torch.sqrt(ratio_tensor[1:])))
        # Divide by 2 to get half height and half width
        anchor_manipulations = torch.stack((-w, -h, w, h)).T.repeat(
            in_height * in_width, 1) / 2

        # Each center point will have `boxes_per_pixel` number of anchor boxes, so
        # generate a grid of all anchor box centers with `boxes_per_pixel` repeats
        out_grid = torch.stack([shift_x, shift_y, shift_x, shift_y],
                               dim=1).repeat_interleave(boxes_per_pixel, dim=0)
        output = out_grid + anchor_manipulations
        return output.unsqueeze(0)

'''
    This model extracts basic features from the image,
    Eventually shrinking the size of image itself to handle 
    lower dimensional images

'''
class BackboneNetwork(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.back_bone = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=32, kernel_size=9),
                                       nn.MaxPool2d(kernel_size=4),
                                       nn.BatchNorm2d(num_features=32),
                                       nn.Conv2d(in_channels=32,out_channels=64,kernel_size=7),
                                       nn.MaxPool2d(kernel_size=4),
                                       nn.BatchNorm2d(num_features=64),
                                       nn.Conv2d(in_channels=64,
                                                 out_channels=128,
                                                 kernel_size=3),
                                       nn.MaxPool2d(kernel_size=4),
                                       nn.BatchNorm2d(num_features=128)
                                       )
    def forward(self, x):
        x = self.back_bone(x)
        return x



class FasterRCNN(pl.LightningModule):

    def __init__(self, lr=1e-4):
        self.save_hyperparameters()
        self.learning_rate = lr
        self.backbone = BackboneNetwork()

    def forward(self, x):

        x = self.backbone(x) ##Features maps

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
    print(bboxes.shape)
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
def test():
    #Get image
    image = Image.open("./test_image.jpg")

    transform = transforms.Compose([
                                    transforms.ToTensor()])

    ## Transform the image in tensor
    image_tensor = transform(image)

    _, img_width, img_height = image_tensor.shape

    ##Get features map from the image
    back_bone = BackboneNetwork()
    image_tensor = image_tensor[None, :, :, :]
    out = back_bone(image_tensor)
    out = out.squeeze()
    n_channel, f_map_width, f_map_height = out.shape

    width_scale_factor = img_width // f_map_width
    height_scale_factor = img_height // f_map_height
    print(width_scale_factor, height_scale_factor)
    rpn = RegionProposalNet()

    bboxes = rpn.generateAnchorBoxes((f_map_width, f_map_height))
    print(bboxes.shape)
    bbox_scale = torch.tensor((width_scale_factor, height_scale_factor, width_scale_factor, height_scale_factor))
    plotBBoxes(bboxes[:, 450:456, :]*bbox_scale, img=image)
    fig = plt.imshow(image)
    show_bboxes(fig.axes, bboxes[250, 250, :, :] * bbox_scale,
                ['s=0.75, r=1', 's=0.5, r=1', 's=0.25, r=1', 's=0.75, r=2',
                 's=0.75, r=0.5'])


if __name__ == "__main__":

    test()