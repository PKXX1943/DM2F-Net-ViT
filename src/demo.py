from PIL import Image
import torch
import torchvision.transforms as transforms
from model_fusion import DM2FNet_fusion
from model import DM2FNet

'''
model = DM2FNet_fusion() 
model.load_state_dict(torch.load('ckpt/RESIDE-fusion/iter_37501_loss_0.01038_lr_0.000041.pth')) 
model.eval()
'''

model = DM2FNet() 
model.load_state_dict(torch.load('ckpt/RESIDE_ITS/iter_40000_loss_0.01128_lr_0.000000.pth')) 
model.eval()


def process_image(image_path):
    image = Image.open(image_path)
    image = image.resize((512, 512))
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    return transform(image)

def model_inference(input_tensor):
    input_tensor = input_tensor.unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
    output_image = output.squeeze(0)
    return output_image
def save_image(tensor, output_path):
    pil_image = transforms.ToPILImage()(tensor)
    pil_image.save(output_path)


input_image_path = 'demo/demo6.png'  # 输入图片路径
output_image_path = 'demo/dehaze6-1.jpg'  # 输出图片路径

input_tensor = process_image(input_image_path)
output_tensor = model_inference(input_tensor)

save_image(output_tensor, output_image_path)

print(f'处理后的图片已保存为：{output_image_path}')
