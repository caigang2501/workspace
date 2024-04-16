import numpy as np
import matplotlib.pyplot as plt
# from machlearning.nsga2_master.example.data import table_p
from example.data import table_p
import pandas as pd

C_WT = 0.019
C_PV = 0.01
C_EB = 0.04
C_HS = 0.023
N_EB = 0.97
T_SH = 0.06
A_CO2 = 0.853
N_SH = 0.87

QEB_MAX = 400
SELL_PRICE = 0.4

def f1(t):
    heat_rest_p = 960
    rest_heats = [heat_rest_p]
    pgt = 0
    for a,b,c,d,e,f,g in table_p[t:t+24]:# '时刻','电负荷','热负荷','冷负荷','风电出力上限','光伏出力上限','外购电价'
        if 1<a<8:
            if a==2:
                ADJUST_QEB_MAX = 0.66*QEB_MAX
                pgt -= ((ADJUST_QEB_MAX-c/N_EB)+(b-(e+f)))*g
                heat_rest_p += ADJUST_QEB_MAX-c/N_EB
            else:
                pgt -= ((QEB_MAX-c/N_EB)+(b-(e+f)))*g
                heat_rest_p += QEB_MAX-c/N_EB
        else:
            if heat_rest_p>=c/N_EB/N:
                heat_rest_p -= c/N_EB
                if e+f>=b:
                    pgt += ((e+f)-b)*SELL_PRICE
                else:
                    pgt -=(b-(e+f))*g
            elif 0<heat_rest_p<c/N_EB:
                if e+f-b-(c/N_EB-heat_rest_p)>0:
                    pgt += (e+f-b-(c/N_EB-heat_rest_p))*SELL_PRICE
                else:
                    pgt -=(b+(c/N_EB-heat_rest_p)-(e+f))*g
                heat_rest_p = 0
            else:
                if e+f>=b:
                    pgt += ((e+f)-b)*SELL_PRICE
                else:
                    pgt -=(b+c/N_EB-(e+f))*g

        heat_rest_p *= (1-T_SH)
        rest_heats.append(heat_rest_p)

    print(pgt,rest_heats[-1])
    function1 = range(len(rest_heats))
    function2 = rest_heats
    plt.xlabel('Function 1', fontsize=15)
    plt.ylabel('Function 2', fontsize=15)
    plt.scatter(function1, function2)
    plt.show()
    return pgt

def f2(t):
    heat_rest_p = 960
    rest_heats = [heat_rest_p]
    pgt = 0
    ans_x = []
    cost_maintain = 0
    cost_elic = 0
    for a,b,c,d,e,f,g in table_p[t:t+24]:# '时刻','电负荷','热负荷','冷负荷','风电出力上限','光伏出力上限','外购电价'
        pwt,ppv,peb,qch,qdis = e,f,0,0,0
        if 1<a<8:
            if a==2:
                ADJUST_QEB_MAX = 0.66*QEB_MAX
                elic_exchange = -((ADJUST_QEB_MAX-c/N_EB)+(b-(e+f)))*g
                pgt += ((ADJUST_QEB_MAX-c/N_EB)+(b-(e+f)))*g
                heat_rest_p += ADJUST_QEB_MAX-c/N_EB

                peb = ADJUST_QEB_MAX
                qch = ADJUST_QEB_MAX-c/N_EB
            else:
                elic_exchange = -((QEB_MAX-c/N_EB)+(b-(e+f)))*g
                pgt += ((QEB_MAX-c/N_EB)+(b-(e+f)))*g
                heat_rest_p += QEB_MAX-c/N_EB

                peb = QEB_MAX
                qch = QEB_MAX-c/N_EB
        else:
            if heat_rest_p>=c/N_EB:
                heat_rest_p -= c/N_EB
                if e+f>=b:
                    elic_exchange = ((e+f)-b)*SELL_PRICE
                    pgt += ((e+f)-b)*SELL_PRICE
                else:
                    elic_exchange = -(b-(e+f))*g
                    pgt +=(b-(e+f))*g
                
                qdis = c/N_EB
            elif 0<heat_rest_p<c/N_EB:
                if e+f-b-(c/N_EB-heat_rest_p)>0:
                    elic_exchange = (e+f-b-(c/N_EB-heat_rest_p))*SELL_PRICE
                    pgt += (e+f-b-(c/N_EB-heat_rest_p))*SELL_PRICE
                else:
                    elic_exchange = -(b+(c/N_EB-heat_rest_p)-(e+f))*g
                    pgt += (b+(c/N_EB-heat_rest_p)-(e+f))*g
                heat_rest_p = 0

                qdis = heat_rest_p
                peb = c/N_EB-heat_rest_p
            else:
                if e+f>=b:
                    elic_exchange = ((e+f)-b)*SELL_PRICE
                    pgt += ((e+f)-b)*SELL_PRICE
                else:
                    elic_exchange = -(b+c/N_EB-(e+f))*g
                    pgt +=(b+c/N_EB-(e+f))*g

                peb = c/N_EB

        heat_rest_p *= (1-T_SH)
        rest_heats.append(heat_rest_p)
        ans_x.append([a,pwt,ppv,peb,qch,qdis,heat_rest_p,elic_exchange])
        cost_maintain += C_WT*pwt + C_PV*ppv + C_EB*peb + (qch+qdis)*C_HS
        cost_elic += b+peb

    pgt -= cost_maintain
    ans_y = (pgt*SELL_PRICE,cost_elic*A_CO2)
    print(pgt,rest_heats[-1],cost_maintain)
    print(ans_x,ans_y)
    df_ansx = pd.DataFrame(ans_x,columns=['time','pwt','ppv','peb','qch','qdis','hhst','pgt'])
    df_ansx.to_csv('machlearning/nsga2_master/example/ans_x.csv')

    function1 = range(len(rest_heats))
    function2 = rest_heats
    plt.xlabel('Function 1', fontsize=15)
    plt.ylabel('Function 2', fontsize=15)
    plt.scatter(function1, function2)
    plt.show()
    return pgt

def f3(t):
    heat_rest_p = 960
    rest_heats = [heat_rest_p]
    pgt = 0
    ans_x = []
    cost_maintain = 0
    cost_elic = 0
    for a,b,c,d,e,f,g in table_p[t:t+24]:# '时刻','电负荷','热负荷','冷负荷','风电出力上限','光伏出力上限','外购电价'
        c *= N_EB
        pwt,ppv,peb,qch,qdis = e,f,0,0,0
        if 1<a<8:
            if a==2:
                ADJUST_QEB_MAX = 0.66*QEB_MAX
                elic_exchange = -((ADJUST_QEB_MAX-c)+(b-(e+f)))*g
                pgt += ((ADJUST_QEB_MAX-c)+(b-(e+f)))*g
                heat_rest_p += (ADJUST_QEB_MAX-c)*N_SH

                peb = ADJUST_QEB_MAX
                qch = ADJUST_QEB_MAX-c
            else:
                elic_exchange = -((QEB_MAX-c)+(b-(e+f)))*g
                pgt += ((QEB_MAX-c)+(b-(e+f)))*g
                heat_rest_p += (QEB_MAX-c)*N_SH

                peb = QEB_MAX
                qch = QEB_MAX-c
        else:
            if heat_rest_p*N_SH>=c:
                heat_rest_p -= c/N_SH
                if e+f>=b:
                    elic_exchange = ((e+f)-b)*SELL_PRICE
                    pgt += ((e+f)-b)*SELL_PRICE
                else:
                    elic_exchange = -(b-(e+f))*g
                    pgt +=(b-(e+f))*g
                
                qdis = c
            elif 0<heat_rest_p*N_SH<c:
                if e+f-b-(c-heat_rest_p*N_SH)>0:
                    elic_exchange = (e+f-b-(c-heat_rest_p*N_SH))*SELL_PRICE
                    pgt += (e+f-b-(c-heat_rest_p*N_SH))*SELL_PRICE
                else:
                    elic_exchange = -(b+(c-heat_rest_p*N_SH)-(e+f))*g
                    pgt += (b+(c-heat_rest_p*N_SH)-(e+f))*g
                heat_rest_p = 0

                qdis = heat_rest_p/N_SH
                peb = c-heat_rest_p*N_SH
            else:
                if e+f>=b:
                    elic_exchange = ((e+f)-b)*SELL_PRICE
                    pgt += ((e+f)-b)*SELL_PRICE
                else:
                    elic_exchange = -(b+c-(e+f))*g
                    pgt +=(b+c-(e+f))*g

                peb = c

        heat_rest_p *= (1-T_SH)
        rest_heats.append(heat_rest_p)
        ans_x.append([a,pwt,ppv,peb,qch,qdis,heat_rest_p,elic_exchange])
        cost_maintain += C_WT*pwt + C_PV*ppv + C_EB*peb + (qch+qdis)*C_HS
        cost_elic += b+peb

    pgt -= cost_maintain
    ans_y = (pgt*SELL_PRICE,cost_elic*A_CO2)
    print(pgt,rest_heats[-1],cost_maintain)
    print(ans_x,ans_y)
    df_ansx = pd.DataFrame(ans_x,columns=['time','pwt','ppv','peb','qch','qdis','hhst','pgt'])
    df_ansx.to_csv('machlearning/nsga2_master/example/ans_x.csv')

    function1 = range(len(rest_heats))
    function2 = rest_heats
    plt.xlabel('Function 1', fontsize=15)
    plt.ylabel('Function 2', fontsize=15)
    plt.scatter(function1, function2)
    plt.show()
    return pgt

def test():
    result = []
    for i in range(1,25):
        result.append(f2(i))
    function1 = range(len(result))
    function2 = result
    plt.xlabel('Function 1', fontsize=15)
    plt.ylabel('Function 2', fontsize=15)
    plt.scatter(function1, function2)
    plt.show()

if __name__=='__main__':
    # test()
    f2(8)





