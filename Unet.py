import torch
import torch.nn as nn

   
########################################################################################################################
def create_upconv(in_channels, out_channels, size=None):
    return nn.Sequential(
        nn.Upsample(size=size, mode='nearest')
        #, nn.ConvTranspose2d(in_channels, out_channels, 2, 2)
        , nn.Conv2d(in_channels,out_channels,3,1,1)
        , nn.BatchNorm2d(num_features=out_channels)
        , nn.GELU()
        , nn.Conv2d(out_channels,out_channels,3,1,1)
        , nn.BatchNorm2d(num_features=out_channels)
        , nn.GELU()
        )

class Unet(nn.Module):
    def __init__(self):
        super().__init__()

        n1 = 32
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.conv_l1 = nn.Sequential(
            nn.Conv2d(1,filters[0],3,1,1)
            , nn.BatchNorm2d(num_features=filters[0])
            , nn.GELU()
            , nn.Conv2d(filters[0],filters[0],3,1,1)
            , nn.BatchNorm2d(num_features=filters[0])
            , nn.GELU()
            )

        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_l2 = nn.Sequential(
            nn.Conv2d(filters[0],filters[1],3,1,1)
            , nn.BatchNorm2d(num_features=filters[1])
            , nn.GELU()
            , nn.Conv2d(filters[1],filters[1],3,1,1)
            , nn.BatchNorm2d(num_features=filters[1])
            , nn.GELU()
            )

        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_l3 = nn.Sequential(
            nn.Conv2d(filters[1],filters[2],3,1,1)
            , nn.BatchNorm2d(num_features=filters[2])
            , nn.GELU()
            , nn.Conv2d(filters[2],filters[2],3,1,1)
            , nn.BatchNorm2d(num_features=filters[2])
            , nn.GELU()
            )

        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_l4 = nn.Sequential(
            nn.Conv2d(filters[2],filters[3],3,1,1)
            , nn.BatchNorm2d(num_features=filters[3])
            , nn.GELU()
            , nn.Conv2d(filters[3],filters[3],3,1,1)
            , nn.BatchNorm2d(num_features=filters[3])
            , nn.GELU()
            )

        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_l5 = nn.Sequential(
            nn.Conv2d(filters[3],filters[4],3,1,1)
            , nn.BatchNorm2d(num_features=filters[4])
            , nn.GELU()
            , nn.Conv2d(filters[4],filters[4],3,1,1)
            , nn.BatchNorm2d(num_features=filters[4])
            , nn.GELU()
            )

        self.deconv_u4 = create_upconv(in_channels=filters[4], out_channels=filters[3], size=(32,32))

        self.conv_u4 = nn.Sequential(
            nn.Conv2d(filters[4],filters[3],3,1,1)
            , nn.BatchNorm2d(num_features=filters[3])
            , nn.GELU()
            , nn.Conv2d(filters[3],filters[3],3,1,1)
            , nn.BatchNorm2d(num_features=filters[3])
            , nn.GELU()
            )

        self.deconv_u3 = create_upconv(in_channels=filters[3], out_channels=filters[2], size=(64,64))

        self.conv_u3 = nn.Sequential(
            nn.Conv2d(filters[3],filters[2],3,1,1)
            , nn.BatchNorm2d(num_features=filters[2])
            , nn.GELU()
            , nn.Conv2d(filters[2],filters[2],3,1,1)
            , nn.BatchNorm2d(num_features=filters[2])
            , nn.GELU()
            )

        self.deconv_u2 = create_upconv(in_channels=filters[2], out_channels=filters[1], size=(128,128))

        self.conv_u2 = nn.Sequential(
            nn.Conv2d(filters[2],filters[1],3,1,1)
            , nn.BatchNorm2d(num_features=filters[1])
            , nn.GELU()
            , nn.Conv2d(filters[1],filters[1],3,1,1)
            , nn.BatchNorm2d(num_features=filters[1])
            , nn.GELU()
            )

        self.deconv_u1 = create_upconv(in_channels=filters[1], out_channels=filters[0], size=(256,256))

        self.conv_u1 = nn.Sequential(
            nn.Conv2d(filters[1],filters[0],3,1,1)
            , nn.BatchNorm2d(num_features=filters[0])
            , nn.GELU()
            , nn.Conv2d(filters[0],filters[0],3,1,1)
            , nn.BatchNorm2d(num_features=filters[0])
            , nn.GELU()
            )

        self.conv1x1_out = nn.Conv2d(filters[0], 1, 1, 1, 0, bias=True)
        self.smout = nn.Sigmoid()
        
    def forward(self, x):

        output1 = self.conv_l1(x)
        input2 = self.maxpool1(output1)
        
        output2 = self.conv_l2(input2)
        input3 = self.maxpool2(output2)
        
        output3 = self.conv_l3(input3)
        input4 = self.maxpool3(output3)
        
        output4 = self.conv_l4(input4)
        input5 = self.maxpool4(output4)
        
        output5 = self.conv_l5(input5)
        input6 = self.deconv_u4(output5)
        
        output6 = self.conv_u4(torch.cat((input6, output4), dim=1))
        input7 = self.deconv_u3(output6)
        
        output7 = self.conv_u3(torch.cat((input7, output3), dim=1))
        input8 = self.deconv_u2(output7)
        
        output8 = self.conv_u2(torch.cat((input8, output2), dim=1))
        input9 = self.deconv_u1(output8)
        
        output9 = self.conv_u1(torch.cat((input9, output1), dim=1))
        out = self.conv1x1_out(output9)
        
        return self.smout(out)
    
if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    a = torch.randn((1,3,256,256)).to(device)
    oper = Unet().to(device)
    b = oper(a)
    print(b.shape)
    
    parameter = list(oper.parameters())

    cnt = 0
    for i in range(len(parameter)):
        cnt += parameter[i].reshape(-1).shape[0]
    
    print(cnt)