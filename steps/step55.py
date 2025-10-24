



def get_conv_outsize(input_size,kernel_size,stride,pad):
    return (input_size+pad*2-kernel_size)//stride+1

H,W=4,4 # input_shape
KH,KW=3,3 # kernel_size
SH,SW=1,1 # stride(垂直方向的步幅，水平方向的步幅)
PH,PW=1,1 # padding(垂直方向的填充，水平方向的填充)

OH=get_conv_outsize(H,KH,SH,PH)
OW=get_conv_outsize(W,KW,SW,PW)

print(OH,OW)

# 卷积运算的过滤器
# (output_channel,input_channel,height,width)

# 小批量的卷积运算的处理流程
# (N,C,H,W)*(OC,C,KH,KW)->(N,OC,OH,OW)+(OC,1,1)->(N,OC,OH,OW)

