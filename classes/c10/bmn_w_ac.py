from torch import nn
from mmaction.models.localizers import BMN
from mmaction.models.builder import LOCALIZERS


@LOCALIZERS.register_module()
class BMNwActionCls(BMN):
    """
    BMN with action classification.
    Adds head for action classificationю.
    <Head training is not implemented>.
    """
    def __init__(self, *args, **kwargs):
        self.num_action_classes = kwargs.pop('num_action_classes')
        super().__init__(*args, **kwargs)
        self.head_action_cls = nn.Sequential(
            nn.Conv1d(self.feat_dim, self.num_action_classes, kernel_size=1),
            nn.Softmax(dim=1)
        )
    
    def forward_test(self, raw_feature, video_meta):
        outputs = super().forward_test(raw_feature, video_meta)        
        action_cls_pred = self.head_action_cls(raw_feature)
        # [batch_size, self.num_action_classes, self.tscale]
        batch_size, nc, ts = action_cls_pred.shape
        
        for b in range(batch_size):
            for p, prop in enumerate(outputs[b]['proposal_list']):
                t_start, t_end = prop['segment']
                proposal_mean_prob = action_cls_pred[b, :, round(t_start):round(t_end)].mean(axis=1)
                outputs[b]['proposal_list'][p]['action_cls'] = torch.argmax(proposal_mean_prob)
        return outputs
