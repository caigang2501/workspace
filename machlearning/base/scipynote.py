from scipy.optimize import minimize
from scipy.integrate import quad,dblquad,nquad

# min = minimize(lambda x:x**2+x+2,0,method="CG")
# print(min)

# integ = quad(lambda x:x,0,2)
# print(integ)

# def bounds_y():
#     return [0, 0.5]
# def bounds_x(y):
#     return [0, 1-2*y]
# area = nquad(lambda x,y:x*y,[bounds_x,bounds_y])
# print(area)