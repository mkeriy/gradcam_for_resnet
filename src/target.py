import torch

def target_id(model, img, transfroms, use_cuda=False):
    model.eval()
    
    img = transfroms(img).unsqueeze(0)
    if use_cuda:
      model = model.cuda()
      img = img.cuda()
    prediction = model(img).squeeze(0).softmax(0)
    return prediction.argmax().item()