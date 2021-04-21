print('hello world')


class TestClass:
    def __init__(self, n1, n2, n3):
        self.n1 = n1
        self.n2 = n2

    def add(self):
        return self.n1 + self.n2


class TestSubClass(TestClass):
    def __init__(self, n1, n2):
        # self.n1 = n1
        # self.n2 = n2
        self.n3 = n1
        self.n4 = n2
        # TestClass.__init__(self, n1=n1, n2=n2)

    def mult(self):
        return self.n3 * self.n4

    def add(self):
        return self.n3 + self.n4

t1 = TestClass(2, 3, 4)
t1.add()
t2 = TestSubClass(3, 4)
# t.mult()

print('hello world')