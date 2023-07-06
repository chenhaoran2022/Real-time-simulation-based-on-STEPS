import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt

A1 = 50 #台风参数
A2 = 40
mu_x = 0
mu_y = 60
sigma1 = 0.45 * 300
sigma2 = 0.05 * 300
 
# 线路段
lines = [  
('L1',-250+30, 80+120,-250+30, 110+120),
('L2_1',-250.0+30, 80.0+120,-250.0+30, 30.0+120),
('L2_2',-250.0+30, 30.0+120,-250+30, 25+120),
('L3_1',-250.0+30, 110.0+120,-214.6+30, 74.6+120),
('L3_2',-214.6+30, 74.6+120,-200+30, 60+120),
('L4_1',-250.0+30, 110.0+120,-207.1+30, 135.7+120),
('L4_2',-207.1+30, 135.7+120,-200+30, 140+120),
('L5_1',-200.0+30, 60.0+120,-200.0+30, 10.0+120),
('L5_2',-200.0+30, 10.0+120,-200.0+30, -40.0+120),
('L5_3',-200.0+30, -40.0+120,-200+30, -60+120),
('L6_1',-200.0+30, 60.0+120,-150.0+30, 60.0+120),
('L6_2',-150.0+30, 60.0+120,-100.0+30, 60.0+120),
('L6_3',-100.0+30, 60.0+120,-80+30, 60+120),
('L7',-200+30, -60+120,-200+30, -110+120),
('L8_1',-200.0+30, -60.0+120,-150.0+30, -60.0+120),
('L8_2',-150.0+30, -60.0+120,-100.0+30, -60.0+120),
('L8_3',-100.0+30, -60.0+120,-50.0+30, -60.0+120),
('L8_4',-50.0+30, -60.0+120,-30+30, -60+120),
('L9',-200+30, -110+120,-200+30, -130+120),
('L10_1',-200.0+30, -110.0+120,-200.0+30, -160.0+120),
('L10_2',-200.0+30, -160.0+120,-200.0+30, -210.0+120),
('L10_3',-200.0+30, -210.0+120,-200+30, -215+120),
('L11',-200+30, -130+120,-200+30, -165+120),
('L12_1',-200.0+30, -130.0+120,-154.6+30, -151.0+120),
('L12_2',-154.6+30, -151.0+120,-135+30, -160+120),
('L13',-200+30, -165+120,-200+30, -215+120),
('L14_1',-200.0+30, -215.0+120,-225.4+30, -171.9+120),
('L14_2',-225.4+30, -171.9+120,-250+30, -130+120),
('L15_1',-250.0+30, -130.0+120,-250.0+30, -80.0+120),
('L15_2',-250.0+30, -80.0+120,-250.0+30, -30.0+120),
('L15_3',-250.0+30, -30.0+120,-250.0+30, 20.0+120),
('L15_4',-250.0+30, 20.0+120,-250+30, 25+120),
('L16',-105+30, -190+120,-135+30, -160+120),
('L17_1',-105.0+30, -190.0+120,-91.6+30, -141.8+120),
('L17_2',-91.6+30, -141.8+120,-80+30, -100+120),
('L18_1',-80.0+30, -100.0+120,-41.0+30, -68.8+120),
('L18_2',-41.0+30, -68.8+120,-30+30, -60+120),
('L19_1',-30.0+30, -60.0+120,-17.89+30, -11.5+120),
('L19_2',-17.9+30, -11.5+120,-15+30, 0+120),
('L20',-15+30, 0+120,0+30, 25+120),
('L21',0+30, 25+120,0+30, 60+120),
('L22_1',0.0+30, 25.0+120,0.0+30, -25.0+120),
('L22_2',0.0+30, -25.0+120,0.0+30, -75.0+120),
('L22_3',0.0+30, -75.0+120,0.0+30, -125.0+120),
('L22_4',0.0+30, -125.0+120,0+30, -150+120),
('L23_1',0.0+30, 25.0+120,48.7+30, 36.4+120),
('L23_2',48.7+30, 36.4+120,97.4+30, 47.7+120),
('L23_3',97.4+30, 47.7+120,146.1+30, 59.1+120),
('L23_4',146.1+30, 59.1+120,150+30, 60+120),
('L24_1',0.0+30, 25.0+120,14.1+30, -23.0+120),
('L24_2',14.1+30, -23.0+120,25+30, -60+120),
('L25_1',0+30,60+120,-50+30,60+120),
('L25_2',-50+30,60+120,-80+30,60+120),
('L26',0+30, 60+120,0+30, 110+120),
('L27_1',150.0+30, 60.0+120,150.0+30, 10.0+120),
('L27_2',150.0+30, 10.0+120,150.0+30, -40.0+120),
('L27_3',150.0+30, -40.0+120,150.0+30, -90.0+120),
('L27_4',150.0+30, -90.0+120,150.0+30, -140.0+120),
('L27_5',150.0+30, -140.0+120,150+30, -160+120),
('L28_1',150.0+30, -160.0+120,114.6+30, -124.6+120),
('L28_2',114.6+30, -124.6+120,100+30, -110+120),
('L29_1',100.0+30, -110.0+120,58.4+30, -82.3+120),
('L29_2',58.4+30, -82.3+120,25+30, -60+120),
('L30_1',-200.0+30, 140.0+120,-150.0+30, 140.0+120),
('L30_2',-150.0+30, 140.0+120,-100.0+30, 140.0+120),
('L30_3',-100.0+30, 140.0+120,-50.0+30, 140.0+120),
('L30_4',-50.0+30, 140.0+120,0+30, 140+120),
('L31',0+30, 140+120,0+30, 110+120),
('L32_1',0.0+30, 140.0+120,50.0+30, 140.0+120),
('L32_2',50.0+30, 140.0+120,60+30, 140+120),
('L33_1',0.0+30, 140.0+120,50.0+30, 140.0+120),
('L33_2',50.0+30, 140.0+120,100.0+30, 140.0+120),
('L33_3',100.0+30, 140.0+120,140+30, 140+120),
('L34_1',60.0+30, 140.0+120,110.0+30, 140.0+120),
('L34_2',110.0+30, 140.0+120,140+30, 140+120)
]

length = []   #线路段长度
for i in range(0,75):
    length_i = ((lines[i][1]-lines[i][3])**2 + (lines[i][2]-lines[i][4])**2)**0.5
    length.append(length_i)

#线路段故障率
segment_probability = []
def integrand(t):   #被积函数
    return length[i]/50 * np.exp((A1 * np.exp(-(((lines[i][1]+lines[i][3])/2 - mu_x + 30/1.414*t)**2 + ((lines[i][2]+lines[i][4])/2 - mu_y - 30/1.414*t)**2)/(2 * sigma1**2)) - 
                                  A2 * np.exp(-(((lines[i][1]+lines[i][3])/2 - mu_x + 30/1.414*t)**2 + ((lines[i][2]+lines[i][4])/2 - mu_y - 30/1.414*t)**2)/(2 * sigma2**2))) *11/30- 18)

#夹角为np.sin(np.pi/2 - np.arctan((60+20*t-(lines[i][2]+lines[i][4])/2)/(abs(lines[i][1]+lines[i][3])/2)) -np.arctan((abs(lines[i][4]-lines[i][2]))/(abs(lines[i][3]-lines[i][1]))))

#求每条线路段在未来1小时内的故障概率P
for i in range(0,75):
    result, error = quad(integrand, 0 ,1)
    Result = 1-np.exp(-result) 
    segment_probability.append(Result) 
    print(str(lines[i][0]) + f"Result: {Result:.10f}")

#由线路段故障概率求整条线路可靠和故障概率
def line_failure_probability():
    line_reliability_probability = []
    line_reliability_probability.append((1-segment_probability[0]))
    line_reliability_probability.append((1-segment_probability[1])*(1-segment_probability[2]))
    line_reliability_probability.append((1-segment_probability[3])*(1-segment_probability[4]))
    line_reliability_probability.append((1-segment_probability[5])*(1-segment_probability[6]))
    line_reliability_probability.append((1-segment_probability[7])*(1-segment_probability[8])*(1-segment_probability[9]))
    line_reliability_probability.append((1-segment_probability[10])*(1-segment_probability[11])*(1-segment_probability[12]))
    line_reliability_probability.append((1-segment_probability[13]))
    line_reliability_probability.append((1-segment_probability[14])*(1-segment_probability[15])*(1-segment_probability[16])*(1-segment_probability[17]))
    line_reliability_probability.append((1-segment_probability[18]))
    line_reliability_probability.append((1-segment_probability[19])*(1-segment_probability[20])*(1-segment_probability[21]))
    line_reliability_probability.append((1-segment_probability[22]))
    line_reliability_probability.append((1-segment_probability[23])*(1-segment_probability[24]))
    line_reliability_probability.append((1-segment_probability[25]))
    line_reliability_probability.append((1-segment_probability[26])*(1-segment_probability[27]))
    line_reliability_probability.append((1-segment_probability[28])*(1-segment_probability[29])*(1-segment_probability[30])*(1-segment_probability[31]))
    line_reliability_probability.append((1-segment_probability[32]))
    line_reliability_probability.append((1-segment_probability[33])*(1-segment_probability[34]))
    line_reliability_probability.append((1-segment_probability[35])*(1-segment_probability[36]))
    line_reliability_probability.append((1-segment_probability[37])*(1-segment_probability[38]))
    line_reliability_probability.append((1-segment_probability[39]))
    line_reliability_probability.append((1-segment_probability[40]))
    line_reliability_probability.append((1-segment_probability[41])*(1-segment_probability[42])*(1-segment_probability[43])*(1-segment_probability[44]))
    line_reliability_probability.append((1-segment_probability[45])*(1-segment_probability[46])*(1-segment_probability[47])*(1-segment_probability[48]))
    line_reliability_probability.append((1-segment_probability[49])*(1-segment_probability[50]))
    line_reliability_probability.append((1-segment_probability[51])*(1-segment_probability[52]))
    line_reliability_probability.append((1-segment_probability[53]))
    line_reliability_probability.append((1-segment_probability[54])*(1-segment_probability[55])*(1-segment_probability[56])*(1-segment_probability[57])*(1-segment_probability[58]))
    line_reliability_probability.append((1-segment_probability[59])*(1-segment_probability[60]))
    line_reliability_probability.append((1-segment_probability[61])*(1-segment_probability[62]))
    line_reliability_probability.append((1-segment_probability[63])*(1-segment_probability[64])*(1-segment_probability[65])*(1-segment_probability[66]))
    line_reliability_probability.append((1-segment_probability[67]))
    line_reliability_probability.append((1-segment_probability[68])*(1-segment_probability[69]))
    line_reliability_probability.append((1-segment_probability[70])*(1-segment_probability[71])*(1-segment_probability[72]))
    line_reliability_probability.append((1-segment_probability[73])*(1-segment_probability[74]))

    line_failure_probability = []
    for i in range(0,34):
        line_failure_probability.append(1-line_reliability_probability[i])
        print('line' + str(i+1) + f" {line_failure_probability[i]:.10f}")
    return line_failure_probability



#PDF
time_array = np.linspace(0, 24, 250)
for t in time_array:
    integral = quad(integrand, 0 ,t)

#CDF
for i in range(0,75):
    segment_cdf_list = []
    for t in time_array:
        integral,error = quad(integrand, 0 ,t)
        segment_probability_t = 1-np.exp(-integral)
    plt.plot(time_array,segment_cdf_list)
plt.show()
