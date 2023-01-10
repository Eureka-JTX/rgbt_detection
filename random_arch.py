from random import choice

archs = [
    [0,1,2],
    [0,1,2,3],
    [0,1],
    [1,2]
]

for i in range(10):
    architecture = []
    # architecture.append(choice(arch) for arch in archs)
    # print(choice(archs[0]))
    for j in range(2):
        a = []
        for arch in archs:
            a.append(choice(arch))
        architecture.append(a)
    print(architecture)