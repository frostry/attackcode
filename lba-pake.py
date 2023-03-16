import math
from gpoly import *
import random
import helper
from ctypes import *
import cProfile
import datetime
from itertools import repeat
from multiprocessing import Pool
import hashlib
from sys import argv
import platform

#j is server i is user
class Ding19(object):
    def __init__(self, n, q, sigma):
        self.n = n
        self.q = q
        self.sigma = sigma
        self.a = Poly.uniform(self.n, self.q)
        self.si = Ding19.gen_secret(self.n, self.q, self.sigma)
        self.ei = Ding19.gen_pubkey_error(self.n, self.q, self.sigma)
        self.s = Ding19.gen_secret(self.n, self.q, self.sigma)
        self.e = Ding19.gen_pubkey_error(self.n, self.q, self.sigma)

    @staticmethod
    def gen_secret(n, q, sigma):
        return Poly.discretegaussian(n, q, sigma)

    @staticmethod
    def gen_pubkey_error(n, q, sigma):
        return Poly.discretegaussian(n, q, sigma)

    @staticmethod
    def gen_shared_error(n, q, sigma):
        return Poly.discretegaussian(n, q, sigma)

    @staticmethod
    def sig(q, v):
        if v > round(q / 4.0) and v < q - math.floor(q / 4):
            return 1
        else:
            return 0

    def signal(self, v):
        return [Ding19.sig(self.q, v[i]) for i in range(self.n)]

    def mod2(self, v, w):
        if self.n != v.n or self.n != len(w): raise Exception
        r = [0 for i in range(self.n)]
        for i in range(self.n):
            r[i] = int(v[i] + w[i] * (self.q - 1) / 2)
            r[i] %= self.q
            r[i] %= 2
        return r
        
    def hash1(self, public):
        seed = hash(str(public))
        random.seed(seed)
        return Poly.discretegaussian(self.n, self.q, self.sigma, random.randrange(2**32))
        # return Poly.discretegaussian(self.n, self.q, self.sigma, seed)



def samp(n, q, sigma):
    time = 0
    for i in range(10000):
        s = datetime.datetime.now()
        ss = Ding19.gen_secret(n, q, sigma)
        e = datetime.datetime.now()
        time += (e - s).microseconds
    return time / 10000

def mul(n, q, sigma):
    time = 0
    a = Poly.uniform(n, q)
    b = Poly.uniform(n, q)
   
    for i in range(10000):
        s = datetime.datetime.now()
        c = a * b
        e = datetime.datetime.now()
        time += (e - s).microseconds
    return time / 10000

def h0(n, q, sigma):
    time = 0
    a = Poly.uniform(n, q)
    
    for i in range(10000):
        s = datetime.datetime.now()
        b = hashlib.sha3_224(str(a).encode('utf-8'))
        e = datetime.datetime.now()
        time += (e - s).microseconds
    return time / 10000

def h1(n, q, sigma):
    time = 0
    execution = Ding19(n, q, sigma)
    public = Poly.uniform(n, q)
    for i in range(10000):
        s = datetime.datetime.now()
        b = execution.hash1(public)
        e = datetime.datetime.now()
        time += (e - s).microseconds
    return time / 10000

def rh2(n, q, l):
        x = Poly(n,q)
        random.seed(l)
        for i in range(n):
            x[i] = random.randrange(q)
        return x

def h2(n, q, sigma):
    time = 0
    a = Poly.uniform(n, q)
    for i in range(10000):
        s = datetime.datetime.now()
        b = rh2(n, q, abs(hash(str(a))))
        e = datetime.datetime.now()
        time += (e - s).microseconds
    return time / 10000


def sig(n, q, sigma):
    time = 0
    execution = Ding19(n, q, sigma)
    public = Poly.uniform(n, q)
    for i in range(10000):
        s = datetime.datetime.now()
        b = execution.signal(public)
        e = datetime.datetime.now()
        time += (e - s).microseconds
    return time / 10000

def mod(n, q, sigma):
    time = 0
    execution = Ding19(n, q, sigma)
    a = Poly.uniform(n, q)
    b = execution.signal(a)
    for i in range(10000):
        s = datetime.datetime.now()
        b = execution.mod2(a, b)
        e = datetime.datetime.now()
        time += (e - s).microseconds
    return time / 10000

if __name__ == "__main__":
    if argv[1] not in ['128','256','512']:
        print("error argv", argv[1])
        exit()
    n = int(argv[1])
    q = 7557773
    sigma = 3.192 

    print("parameters: n = {:d}, q = {:d}, sigma = {:f}".format(n, q, sigma))

    for seed in range(1):

        random.seed(seed + time.time())
        a = Poly.uniform(n, q)
        global execution
        
        print("10000 exec")
        
        h0_t = h0(n,q,sigma)
        print("hash0 is ", h0_t)
        h1_t = h1(n,q,sigma)
        print("hash1 is ", h1_t)
        h2_t = h2(n,q,sigma)
        print("hash2 is ", h2_t)
        samp_t = samp(n,q,sigma)
        print("samp is ", samp_t)
        mul_t = mul(n,q,sigma)
        print("mul is ", mul_t)
        sig_t = sig(n,q,sigma)
        print("sig is ", sig_t)
        mod_t = mod(n,q,sigma)
        print("mod is ", mod_t)


        if platform.machine() == 'aarch64':
            alltime = 6 * h0_t + 2 * h1_t + h2_t + 3 * samp_t + 4 * mul_t + sig_t + 2 * mod_t
            print("time of simulation on user: ", alltime)

        elif platform.machine() == 'x86_64':
            alltime = 5 * h0_t + 2 * h1_t + 3 * samp_t + 4 * mul_t + sig_t + 2 * mod_t
            print("time of simulation on server: ", alltime)
