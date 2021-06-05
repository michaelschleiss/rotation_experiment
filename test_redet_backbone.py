import torch
from torchsummary import summary


from backbone import ReResNet
CHECKPOINT_PATH = 'checkpoints/re_resnet50_c8_batch256-25b16846.pth'
CHECKPOINT = torch.load(CHECKPOINT_PATH)['state_dict']


model = ReResNet(depth=50)
model.load_state_dict(CHECKPOINT, strict = False)
model.eval()
model.cuda()

inputs = torch.rand(1, 3, 640, 480).cuda()
res = model.forward(inputs)
#summary(model, (3, 640, 480))

print(res[0].shape)
