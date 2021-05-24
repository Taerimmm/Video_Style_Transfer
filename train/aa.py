class Array:
    def __init__(self):
        self.arr = [1,2,3,4,5,6,7,8]
    def __getitem__(self,key):
        return self.arr[key]
a = Array()
print(a)
for i in a:
    print(i, end = " ")