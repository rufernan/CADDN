from model import CADDN
from smp.data import *
from smp.dice import *
from smp.functional import *
from smp.functional import compute_metrics
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import piq
import torch



print('Loading the data...')
(xtra, ytra, xtst, ytst, xval, yval) = load_AISD('./data/AISD')



NUM_CHANNELS = xtra.shape[1]
OUT_CLASSES = np.unique(np.rint(ytra)).shape[0]-1
DROP = 0.1

seg_model = CADDN(in_channels=NUM_CHANNELS, classes=OUT_CLASSES, imgage_decoder_dropout=DROP)
seg_loss = DiceLoss()
img_loss = piq.SSIMLoss(data_range=1., reduction='mean')



GPU = 0
BATCH_SIZE = 1
EPOCHS = 100
EPOCHS = 3
LEARNING_RATE = 0.001
LR_DECAY = 0.5
PATIENCE = 20
ALPHA = 1
BETA = 1

tensor_xtra = torch.Tensor(xtra.astype(np.float32))
tensor_ytra = torch.Tensor(ytra.astype(np.float32))
tradata = TensorDataset(tensor_xtra,tensor_ytra)
traloader = DataLoader(dataset=tradata,batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
tensor_xval = torch.Tensor(xval.astype(np.float32))
tensor_yval = torch.Tensor(yval.astype(np.float32))
valdata = TensorDataset(tensor_xval,tensor_yval)
valloader = DataLoader(dataset=valdata,batch_size=1, shuffle=False)

torch.backends.cudnn.benchmark = True
device = torch.device('cuda:'+str(GPU) if torch.cuda.is_available() else 'cpu')

model = seg_model.to(device)
criterion = seg_loss.to(device)
criterion2 = img_loss.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
scheduler = StepLR(optimizer, step_size=PATIENCE, gamma=LR_DECAY)



print('Training the model...')
for epoch in range(EPOCHS):
    
    model.train()
    epoch_loss = 0
    epoch_shadow_loss = 0
    epoch_image_loss = 0
    epoch_iou = 0
    epoch_f1 = 0
    epoch_ber = 0
    
    for iteration, batch in enumerate(traloader):
        input, target = batch[0].to(device), batch[1].to(device)
        prediction, image = model(input)
        shadow_loss = criterion(prediction, target)
        image_loss = criterion2(torch.clamp(image,0,1), input)
        loss = ALPHA * shadow_loss + BETA * image_loss
        loss.backward(retain_graph=True)
        optimizer.step()
        optimizer.zero_grad()
        epoch_loss += loss.item()
        epoch_shadow_loss += shadow_loss.item()
        epoch_image_loss += image_loss.item()
        pred_mask = prediction
        all_metrics = compute_metrics(pred_mask, target.int())
        epoch_iou += all_metrics[0].item()
        epoch_f1 += all_metrics[1].item()
        epoch_ber += all_metrics[5].item()
        
    model.eval() 
    val_loss = 0
    val_iou = 0
    val_f1 = 0
    val_ber = 0
    val_iou_f1 = 0

    with torch.no_grad():
        for index, batch in enumerate(valloader):
            input, target = batch[0].to(device), batch[1].to(device)
            prediction, image = model(input)
            shadow_val_loss = criterion(prediction, target).item()
            image_val_loss = criterion2(torch.clamp(image,0,1), input).item()
            val_loss += ALPHA * shadow_val_loss + BETA * image_val_loss
            pred_mask = prediction
            all_metrics = compute_metrics(pred_mask, target.int())
            val_iou += all_metrics[0].item()
            val_f1 += all_metrics[1].item()
            val_ber += all_metrics[5].item()
            val_iou_f1 += all_metrics[0].item() + all_metrics[1].item()
            
    model.train()
    scheduler.step()

    str_loss = "{:8.6f} ({:6.4f}, {:6.4f})".format(np.sum(epoch_loss) / len(traloader), np.sum(epoch_shadow_loss) / len(traloader), np.sum(epoch_image_loss) / len(traloader))
    epoch_metrics = "(IoU {:5.4f}, F1 {:5.4f}, BER {:5.4f})".format( np.sum(epoch_iou)/len(traloader), np.sum(epoch_f1)/len(traloader), np.sum(epoch_ber)/len(traloader) )
    val_metrics = "(IoU {:5.4f}, F1 {:5.4f}, BER {:5.4f})".format( np.sum(val_iou)/len(valloader), np.sum(val_f1)/len(valloader), np.sum(val_ber)/len(valloader))
    print( "--Epoch " + "{:>3}".format( str(epoch+1)) + ". Loss: "+ str_loss + ", Tra " + epoch_metrics + ", Val " + val_metrics, end='')
    print("\n", end='')


torch.save(model.state_dict(), "model.pth" )
print('Final model saved!')
