import sys,os
sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from example.data import table_p
from nsga2_gp.problem import Problem
from nsga2_gp.evolution import Evolution
from nsga2_gp.individual import Individual
from nsga2_gp.utils import adjust
import matplotlib.pyplot as plt
import pandas as pd

C_WT = 0.019
C_PV = 0.01
C_EB = 0.04
C_HS = 0.023

N_EB = 0.97
N_SH = 0.87

SELL_PRICE = 0.4

def calculate(individual:Individual):
    cost_money = 0
    i = 0
    ans = []
    heat_rest = 960
    for peb in individual.features:
        a,b,c,d,e,f,g = table_p[i]# '时刻','电负荷','热负荷','冷负荷','风电出力上限','光伏出力上限','外购电价'
        pwt,ppv,qch,qdis = e,f,0,0
        if peb*N_EB-c>0:
            qch = peb*N_EB-c
            heat_rest += qch*N_SH
        else:
            qdis = c - peb*N_EB
            heat_rest -= qdis/N_SH
        elic_exchange = pwt+ppv-(b+peb)

        if pwt+ppv-(b+peb)>0:
            cost_money -= (pwt+ppv-(b+peb))*SELL_PRICE
        else:
            cost_money += ((b+peb)-(pwt+ppv))*g
        cost_money += pwt*C_WT + ppv*C_PV + peb*C_EB + max(qch,qdis)*C_HS
        ans.append([a,pwt,ppv,peb,qch,qdis,heat_rest,elic_exchange])
        i += 1
    individual.ans = ans
    return cost_money

def f1(individual:Individual):
    calculate(individual)
    adjust(individual)
    return calculate(individual)

def f2(individual:Individual):
    cost_elic = 0
    i = 0
    for peb in individual.features:
        a,b,c,d,e,f,g = table_p[i]
        cost_elic += peb+b
        i += 1
    return cost_elic


def solve():
    save_path = 'machlearning/nsga2_master/example/result/'
    problem = Problem(objectives=[f1, f2], num_of_variables=24, variables_range=[(0,3)])
    evo = Evolution(problem,10000,15)
    evol = evo.evolve()
    with pd.ExcelWriter(f'{save_path}ans.xlsx') as writer:
        for i,individral in enumerate(evol):
            df_ansx = pd.DataFrame(individral.ans,columns=['time','pwt','ppv','peb','qch','qdis','hhst','pgt'])
            # df_ansx.to_csv(f'{save_path}ans_{i}.csv',index=False)
            name = ''.join([str(round(cost)) for cost in individral.objectives])
            df_ansx.to_excel(writer, sheet_name=name, index=False)
    func = [i.objectives for i in evol]

    function1 = [i[0] for i in func]
    function2 = [i[1] for i in func]
    plt.xlabel('cost_money', fontsize=15)
    plt.ylabel('cost_elic', fontsize=15)
    plt.scatter(function1, function2)
    plt.savefig(f'{save_path}ans.png')
    plt.show()

if __name__=='__main__':
    solve()





