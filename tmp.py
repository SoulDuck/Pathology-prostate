#-*- coding:utf-8 -*-
class A(object):
    def __init__(self):
        self.a_1 =3

# 클래스 A의 self변수들은 class B의 메모리안에 있다
class B(A):
    def __init__(self):
        super(B ,self).__init__()
        print self.a_1


if __name__ == '__main__':
    b=B()
