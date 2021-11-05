import math
from gpoly import *
import random
from ctypes import *
import cProfile
import time
from itertools import repeat
from multiprocessing import Pool
from sys import argv

class Ding19(object):
    def __init__(self, n, q, sigma, seed = None):
        self.n = n
        self.q = q
        self.sigma = sigma
        self.a = Poly.uniform(self.n, self.q)
        self.si = Ding19.gen_secret(self.n, self.q, self.sigma)
        self.ei = Ding19.gen_pubkey_error(self.n, self.q, self.sigma)
        self.sj = Ding19.gen_secret(self.n, self.q, self.sigma, seed)
        self.ej = Ding19.gen_pubkey_error(self.n, self.q, self.sigma)

    @staticmethod
    def gen_secret(n, q, sigma, seed = None):
        return Poly.discretegaussian(n, q, sigma, seed)

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
        
    def hash1(self, Pa):
        seed = hash(str(Pa))
        random.seed(seed)
        return Poly.discretegaussian(self.n, self.q, self.sigma, random.randrange(2**32))
        
    def hash2(self, xs):
        seed = hash(str(xs))
        random.seed(seed)
        return Poly.discretegaussian(self.n, self.q, self.sigma, random.randrange(2**32))

    def alice_init(self):
        self.xi = self.a * self.si + 2 * self.ei
        return self.xi

    #def bob(self, Pi, Pj, xi, simplified=True):
    def bob(self, Pa):
        self.Pb = self.a * self.sj + 2 * self.ej
        es = Ding19.gen_pubkey_error(self.n, self.q, self.sigma)
        rs = Ding19.gen_secret(self.n, self.q, self.sigma, secrets.randbits(32))
        self.xs = self.a * rs + 2 * es  
        #print(self.xs)
        c = self.hash1(Pa)
        d = self.hash2(self.xs)
        
        self.eb = Ding19.gen_shared_error(self.n, self.q, self.sigma)

        Pabar = Pa + self.a * c 
        self.kb = Pabar * (self.sj + d) + 2 * self.eb
    
        self.wj = self.signal(self.kb)
        self.skj = self.mod2(self.kb, self.wj)
        return (self.Pb, self.xs, self.wj, self.skj)
        
    # def alice_resp(self, Pi, Pj, yj, wj):
    #     c = self.hash1(Pi, Pj, self.xi)
    #     d = self.hash2(Pi, Pj, self.xi, yj)
    #     self.fi = Ding19.gen_shared_error(self.n, self.q, self.sigma)
    #     yjbar = yj + self.a * d + 2 * self.fi
    #     self.gi = Ding19.gen_shared_error(self.n, self.q, self.sigma)
    #     self.ki = yjbar * (self.si + c) + 2 * self.gi
    #     self.ski = self.mod2(self.ki, wj)
    #     return self.ski

def get_zeros(coeffs, n):
    zero_count = 0
    temp_count = 0
    for i in range(n):
        if coeffs[i] == 0:
            temp_count += 1
        else:
            if temp_count > zero_count:
                zero_count = temp_count
            temp_count = 0
    return zero_count

def process_k(k, n, kn_bnd, t):
    global execution
    global f

    signals = [None for i in range(n)]
    query = 0

    a = execution.a
    Pa = k * f
    # print(k)
    got_signals = 0
    while got_signals < n:
        (Pb, xs, wj, skj) = execution.bob(Pa)
        c = execution.hash1(Pa)
        d = execution.hash2(xs)
        known = c * Pb + a * c * d + d * Pa
        query += 1
        for i in range(n):
            if (known[i] <= kn_bnd  or known[i] >= (q - kn_bnd)) and signals[i] == None:
                signals[i] = wj[i]
                got_signals += 1
                # print(got_signals)
    # print("finish:", k)
    # print(signals)
    result = [signals, query]
    return result


def collect_signals(n, q, istar, kn_bnd, t):
    global f
    f[0] = 1
    f[istar] = 1
    all_k = []
    maxs = 11
    if istar != 0:
        maxs *= 2

    temp = 1
    while temp * t < int(q/2 + t):
        all_k.append(temp * t)
        temp += 1
    # print(t)
    # print(all_k)

    # print(round(q / 4))
    # print((q - math.floor(q / 4)),"\n")
    stable_k = []
  
    # s_all = [3,11]
    for k in all_k:
        flag = True
        # print('for k ',k,':')
        ks = (k * maxs) % q
        # print(ks-kn_bnd, " ", ks, " ", ks + kn_bnd)
        if abs(ks - round(q / 4)) < kn_bnd or abs(ks - (q - math.floor(q / 4))) < kn_bnd:
                flag = False
        if flag == True:
            stable_k.append(k)
    # print(stable_k)
    
    stable_t = int(q/(2 * maxs) - 2 * kn_bnd)
    # print(stable_t)
    final_k = []
    i = 0
    while i+1 < len(stable_k):
        if stable_k[i+1] - stable_k[i] < stable_t:
            final_k.append(stable_k[i])
            i += 2
        else:
            final_k.append(stable_k[i])
            i += 1
    # if stable_k[len(stable_k) - 1] not in final_k:
    #     final_k.append(stable_k[len(stable_k) - 1])
    final_k.append(int(q/2))
    # print(final_k)
    signal_number = len(final_k)
    # exit()

    # final_k = all_k
    # print(final_k)
    # signal_number = len(final_k)
    # results = process_k(all_k[2], n, kn_bnd, t)
    # print(results)
    # exit()

    pool = Pool(signal_number)
    results = pool.starmap(process_k, zip(final_k, repeat(n), repeat(kn_bnd), repeat(t)))
    pool.close()
    pool.join()

    signals = []
    query = []
    for i in results:
        signals.append(i[0])
        query.append(i[1])
    # print('signals :\n', signals)
    f[istar] = 0
    return signals, query, signal_number

def count_changes(f):
    curr = 0
    count = 0
    for i in range(len(f)):
        if f[i] != curr: 
            count += 1
            curr = f[i]
    return count 

def collect_abs(n, q, sigma, istar, kn_bnd, t):
    signals, query, signal_number = collect_signals(n, q, istar, kn_bnd, t)
    print(sum(query)," ", signal_number)
    coeffs_abs = list(range(n))
    # print(count_changes([0,1,0,1]))
    # return coeffs_abs, sum(query)
    for i in range(n):
        f = [signals[k][i] for k in range(signal_number)]
        # print(f)
        coeffs_abs[i] = count_changes(f)
        if istar == 0 and coeffs_abs[i] != abs(execution.sj[i]):
            print("error: ", coeffs_abs[i], execution.sj[i], f)
    return coeffs_abs, sum(query)

def attack_step3(n, q, coeffs, MAX_ISTAR):
    sign_comp = [[None for j in range(n)] for istar in range(n)]
    for istar in range(1, MAX_ISTAR):
        for i in range(istar, n):
            if coeffs[0][i] != 0 and coeffs[0][i - istar] != 0:
                if coeffs[istar][i] == coeffs[0][i] + coeffs[0][i - istar]: sign_comp[i][i - istar] = "S"
                else: sign_comp[i][i - istar] = "D"
        for i in range(istar):
            if coeffs[0][i] != 0 and coeffs[0][(i - istar) % n] != 0:
                if coeffs[istar][i] == coeffs[0][i] - coeffs[0][(i - istar) % n]:
                    sign_comp[i][(i - istar) % n] = "S"
                else:
                    sign_comp[i][(i - istar) % n] = "D"
    coeffs_signed = Poly(n, q)
    for i in range(n): coeffs_signed[i] = coeffs[0][i]
    for i in range(0, n):
        if coeffs_signed[i] != 0:
            pos_votes = 0
            neg_votes = 0
            for j in range(0, i):
                if coeffs_signed[j] < 0:
                    if sign_comp[i][j] == "S": neg_votes += 1
                    if sign_comp[i][j] == "D": pos_votes += 1
                elif coeffs_signed[j] > 0:
                    if sign_comp[i][j] == "S": pos_votes += 1
                    if sign_comp[i][j] == "D": neg_votes += 1
            if neg_votes > pos_votes: coeffs_signed[i] *= -1
    return coeffs_signed

def attack(n, q, sigma, a, sj, kn_bnd, kn_bnd2, t, t2):
    query = 0
    coeffs_abs, getquery = collect_abs(n, q, sigma, 0, kn_bnd, t)
    query += getquery
    print(coeffs_abs)
    # return True, query
    # print("                           ", strarray(range(n)))
    # print("istar = ", 0, "coeffs[istar] = ", strarray(coeffs_abs))
    zero_count = get_zeros(coeffs_abs,n) #3
    # print(zero_count)
    MAX_ISTAR = zero_count + 2
    coeffs = list(range(MAX_ISTAR))
    coeffs[0] = coeffs_abs
    for istar in range(1,MAX_ISTAR):
        coeffs[istar], getquery = collect_abs(n, q, sigma, istar, kn_bnd2, t2)
        print("istar = ", istar, "coeffs[istar] = ", coeffs[istar])
        query += getquery

    coeffs_signed = attack_step3(n, q, coeffs, MAX_ISTAR)
    print("sB            = ", strarray(sj))
    print("coeffs_signed = ", strarray(coeffs_signed))
    return sj == coeffs_signed or sj == -1 * coeffs_signed, query

def strarray(a):
    s = "["
    for x in a:
        if isinstance(x, bool): s += " T" if x else " F"
        elif isinstance(x, int): s += "{:2d}".format(x)
        elif isinstance(x, str): s += "{:2s}".format(x)
        elif x is None: s += "  "
        else: assert False, "Invalid type " + x
        s += " "
    s += "]"
    return s

if __name__ == "__main__":
    if argv[1] not in ['128','256','512']:
        print("error argv", argv[1])
        exit()
    n = int(argv[1])
    q = 7557773
    sigma = 3.192 
    kn_bnd = 82000
    kn_bnd2 = 40000
    t1 = 170000
    t2 = 86000
    print("parameters: n = {:d}, q = {:d}, sigma = {:f}".format(n, q, sigma))


    c = 1
    alltime = 0
    allq = 0
    succ = 0
    now = int(time.time())
    for seed in range(c):
        start = time.time()
        print("======================")
        print("count = ", seed)
        random.seed(seed + now)
        seed += now
        # seed = 2
        a = Poly.uniform(n, q)
        global f, execution
        f = Poly(n,q)
        execution = Ding19(n, q, sigma, seed)
        # execution.sj[0] = 11
        sB = execution.sj
        #print("     ", strarray(range(n)))
        print(min(sB), max(sB), list(sB).index(max(sB)))
        print("sB = ", strarray(sB))
        # eb = execution.ej
        # print(eb, max(eb))
        # fp = open('drel512', 'a+')
        # fp.write("sB is " + str(list(sB)) + "\n")
        # fp.close()
        rel, getquery = attack(n,q,sigma,a,sB,kn_bnd,kn_bnd2,t1,t2)
        if rel != False:
            succ += 1
            print("queries: ", getquery)
        else:
            print("False")
        end = time.time()
        t = end - start
        alltime += t
        allq += getquery
        hours, rem = divmod(t, 3600)
        minutes, seconds = divmod(rem, 60)
        print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))
    print("average: {} {} {}".format(allq / c, alltime / c, succ / c))