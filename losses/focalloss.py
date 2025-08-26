# --------------------------------------------------------
# 
# Author: Ali Ghelmani,       Created: December 12, 2022
# --------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class FocalLoss(nn.Module):
    r"""
        This is an implemenation of Focal Loss, proposed in the Focal Loss for Dense Object Detection paper.
            
            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])
    
        The losses are averaged across observations for each minibatch.

        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this loss
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5), 
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field reduction is set to 'sum', the losses are
                                instead summed for each minibatch.

    """
    def __init__(self, class_num, alpha=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        if alpha is None:
            # According to RetinaNet: alpha=0.25 works best with gamma=2 and 1-alpha should be assigned to background
            self.alpha = Variable(0.25 * torch.ones(1, class_num))
        else:
            self.alpha = alpha if isinstance(alpha, Variable) else Variable(alpha)

        self.gamma = gamma
        self.class_num = class_num
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, label: torch.Tensor):
        """ The pred and labels are assumed to be of the size (#batch, anchor*nW*nH, #cls)

        Args:
            pred (_type_): torch.Tensor of size [n_batch, anchor*h*w, n_cls]
            label (_type_): torch.Tensor of size [n_batch, anchor*h*w, n_cls]
        """
        log_prob = F.log_softmax(pred, dim=-1)
        mid_fl = -1.0 * ((1 - torch.exp(log_prob)) ** self.gamma) * log_prob
        
        alpha = self.alpha.to(pred.device)
        one_hot = label * alpha
        fl = (one_hot * mid_fl).sum()
        
        #---------------------------------------------------------------------------------#
        # If reduction is 'mean' the normalization is done based on the number of anchors #
        # assigned to the objects and the background cases are ignored, since their loss  #
        # is negligible (RetinaNet).                                                      #
        #---------------------------------------------------------------------------------#
        loss = fl / label.sum().item() if self.reduction == 'mean' else fl
        
        return loss