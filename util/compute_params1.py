
import math
# 已经通过 from config.params import get_params 引入了 get_params
# params


'''communication'''
# distance (sampled uniformly from [100, 200] m)
adress = [(500, 700), (300, 400), (800, 200),(0, 0)]

# 计算第一个点到其它位置的距离
dis = [math.hypot(adress[0][0] - x, adress[0][1] - y) for (x, y) in adress[1:]]
print(dis)

# bandwidth (Hz)
W = 10 * pow(10, 6) / 3
# noise power (mW)
sigma2 =pow(10, -174 / 10) * W
# path loss
def path_loss(dis):
    return pow(10, -128.1 / 10) * pow((dis / 1000), -3.76)

losses = [path_loss(d) for d in dis]
print(f"Path losses: {losses}")

# [1] A Joint Uplink/Downlink Resource Allocation Algorithm in OFDMA Wireless Networks
# [2] Energy-Efficient Resource Allocation for Mobile cld Computing-Based Augmented Reality Applications
# [3] On the Application of Uplink/Downlink Decoupled Access in Heterogeneous Mobile cld Computing