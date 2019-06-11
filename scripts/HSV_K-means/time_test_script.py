from timeit import Timer

t = []

for i in range(5):
    t1 = Timer("main()", "from HSV import main").timeit(number=1)
    t.append(t1)

print("test ", sum(t)/len(t), "seconds")
