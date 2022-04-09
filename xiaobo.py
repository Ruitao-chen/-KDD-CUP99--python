import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fftpack import fft,ifft
import math
import pywt
import csv
from pandas import DataFrame;

data = pd.read_csv('E:/python/shujuwajue/kddcup.data_10_percent/kddcup_data_10_percent_corrected.csv',encoding="gbk")
print(data.columns)
y_values = np.array(data['0.1'])
x_values = np.array(data['0'])
plt.plot(x_values,y_values)
plt.show()

yy=fft(y_values)                     #快速傅里叶变换
yreal = yy.real               # 获取实数部分
yimag = yy.imag               # 获取虚数部分

yf=abs(fft(y_values))                # 取模
yf1=abs(fft(y_values))/((len(x_values)/2))           #归一化处理
yf2 = yf1[range(int(len(x_values)/2))]  #由于对称性，只取一半区间

xf = np.arange(len(y_values))        # 取频率
xf1 = xf
xf2 = xf[range(int(len(x_values)/2))]  #取一半区间

#plt.plot(xf,yf,'r') #显示原始信号的FFT模值
plt.subplot(2, 1, 1)
plt.plot(xf2,yf2,'g')
#plt.show()
#-----------------------分界线-------------------------------------------
w = pywt.Wavelet('sym8')#选用sym8小波
maxlev = pywt.dwt_max_level(len(data), w)#最大分解级别，返回max_level。db.dec_lenx为小波的长度
coeffs = pywt.wavedec(y_values, w, mode='constant',level=maxlev)#分解波，constant是边缘都填充0，不管tensor的内容
threshold = 0.4
#plt.figure()
for i in range(1, len(coeffs)):
    coeffs[i] = pywt.threshold(coeffs[i], threshold*max(coeffs[i]))  # 将噪声滤波

datarec = pywt.waverec(coeffs, 'sym8')  # 将信号进行小波重构


yy1=fft(datarec)
yreal = yy1.real               # 获取实数部分
yimag = yy1.imag               # 获取虚数部分

yf_1 = abs(fft(datarec))                # 取模
yf1_1 = abs(fft(datarec))/((len(x_values)/2))           #归一化处理
yf2_1 = yf1_1[range(int(len(x_values)/2))]  #由于对称性，只取一半区间

xf_1 = np.arange(len(datarec))        # 频率
xf1_1 = xf_1
xf2_1 = xf_1[range(int(len(x_values)/2))]  #取一半区间

plt.subplot(2, 1, 2)
plt.plot(xf2_1,yf2_1)

#plt.plot(datarec,'g')
plt.show()
#d = DataFrame(datarec)
#d.to_csv('C:/Users/ASUS/Desktop/data.csv')
df1 = pd.DataFrame(datarec)
df1.to_excel("E:/python/shujuwajue/kddcup.data_10_percent/output.xlsx")  
