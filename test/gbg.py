a = ['3','3 ',3,3.126,3.124,'']
def dealdata(d):
    if d:
        # print("%.2f" %float(d))
        if type(d) is str:
            # print('len:',len(d))
            d.strip()
            # print('striplen:',len(d))
        a = float("{:.2f}".format(float(d)))
        a = "%.2f" %float(d)