

with open("validate.txt","w") as f:

    for i in range(25):
        f.writelines("{0} {1} {2}\n".format(0,0,0^0))
    for i in range(25):
        f.writelines("{0} {1} {2}\n".format(0, 1, 0 ^ 1))
    for i in range(25):
        f.writelines("{0} {1} {2}\n".format(1, 0, 1 ^ 0))
    for i in range(25):
        f.writelines("{0} {1} {2}\n".format(1, 1, 1 ^ 1))
