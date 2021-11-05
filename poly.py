from ctypes import *
import math
import random
import binascii
import time

c_poly = CDLL('./c_poly.so')
 
def listequals(a, b):
    if len(a) != len(b): return False
    for i in range(len(a)):
        if a[i] != b[i]:
            return False
    return True

class Poly(object):
    def __init__(self, n, q, coeffs = None):
        self.n = n
        self.q = q
        if coeffs is None:
            data = [0 for i in range(n)]
            self.coeffs = (c_int64 * n)(*data)
        else:
            if len(coeffs) != n: raise ValueError
            self.coeffs = coeffs

    def dupe(self):
        r = Poly(self.n,self.q)
        r.coeffs = (c_int64 * self.n)(*[self.coeffs[i] for i in range(self.n)])
        return r

    def canonicalize_q(self):
        for i in range(self.n):
            self[i] = (self[i] % self.q)

    def canonicalize_centred(self):
        self.canonicalize_q()
        for i in range(self.n):
            if self[i] > (self.q - 1) / 2:
                self[i] = self[i] - self.q

    def __str__(self):
        # self.canonicalize_centred()
        return str([self.coeffs[i] for i in range(self.n)])
    
    def __eq__(self, other):
        self.canonicalize_q()
        other.canonicalize_q()
        return self.n == other.n and self.q == other.q and listequals(self.coeffs, other.coeffs)

    def __getitem__(self, i):
        return self.coeffs[i]
    
    def __setitem__(self, i, v):
        self.coeffs[i] = v

    def addpy(self, other):
        if self.n != other.n or self.q != other.q: raise Exception
        r = Poly(self.n, self.q)
        for i in range(self.n):
            r[i] = (self[i] + other[i]) % self.q
        return r

    def addc(self, other):
        if self.n != other.n or self.q != other.q: raise Exception
        r = Poly(self.n,self.q)
        c_poly.poly_add(byref(r.coeffs), byref(self.coeffs), byref(other.coeffs), self.n)
        return r

    def __add__(self, other):
        return self.addc(other)

    def mulpy(self, other):
        r = Poly(self.n, self.q)
        if isinstance(other, int):
            for i in range(self.n):
                r[i] = (self[i] * other) % self.q
            return r
        else:
            if self.n != other.n or self.q != other.q: raise Exception
            for i in range(self.n):
                for j in range(self.n):
                    r[(i + j) % self.n] += (self[i] * other[j]) * (1 if i + j < self.n else -1)
                    r[(i + j) % self.n] %= self.q
            return r

    def mulcmont(self, other):
        r = Poly(self.n, self.q)
        if isinstance(other, int):
            c_poly.poly_mul_int(byref(r.coeffs), byref(self.coeffs), c_int64(other), self.n)
        else:
            c_poly.poly_mul_mont(byref(r.coeffs), byref(self.coeffs), byref(other.coeffs), self.n)
        return r

    def __mul__(self, other):
        return self.mulcmont(other)
    
    def __rmul__(self, other):
        return self * other

    @staticmethod
    def uniform(n, q, rng = random):
        x = Poly(n,q)
        for i in range(n):
            x[i] = rng.randrange(q)
        return x
    
    @staticmethod
    def discretegaussian(n, q, gamma, seed = None): # samples random polynomial with each coefficient from a discrete Gaussian of parameter gamma; caller needs to scale gamma appropriately based on number of coefficients
        x = Poly(n,q)
        if seed:
            c_poly.init_rand(seed)
        else:
            seed = str(random.randrange(2 ** 32))
            c_poly.init_rand(seed)
        for i in range(n):
            x[i] = c_poly.sample_from_cdf()
        return x
    
    @staticmethod
    def zero(n, q):
        x = Poly(n,q)
        return x

c_poly.init_rand(0)

if __name__ == "__main__":
    NUM_TESTS = 500
    n = 512
    q = 7557773
    # Addition tests
    # for i in range(NUM_TESTS):
    #     a = Poly.uniform(n, q)
    #     b = Poly.uniform(n, q)
    #     c1 = a.addpy(b)
    #     c2 = a.addc(b)
    #     if c1 != c2:
    #         print("MISMATCH:")
    #         print("a                 = {:s}".format(str(a)))
    #         print("b                 = {:s}".format(str(b)))
    #         print("a + b (in Python) = {:s}".format(str(c1)))
    #         print("a + b (in C)      = {:s}".format(str(c2)))
    #         assert False
    # print("Addition tests passed")
    # Polynomial multiplication tests
    start = time.time()
    for i in range(NUM_TESTS):
        a = Poly.uniform(n, q)
        b = Poly.uniform(n, q)
        c1 = a.mulpy(b)
        c2 = a.mulcmont(b)
        if c1 != c2:
            print("MISMATCH:")
            print("a                 = {:s}".format(str(a)))
            print("b                 = {:s}".format(str(b)))
            print("a * b (in Python) = {:s}".format(str(c1)))
            print("a * b (in C)      = {:s}".format(str(c2)))
            assert False
    end = time.time()
    print("Polynomial multiplication tests passed", end - start)


