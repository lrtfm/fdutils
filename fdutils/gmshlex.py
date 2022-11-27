import numpy as np

def SN1(p):
    return p + 1

def SN2(p):
    return SN1(p) * SN1(p + 1) // 2
    
def SN3(p): 
    return SN2(p) * SN1(p + 2) // 3
    
def SI1(p, i):
    return i

def SI2(p, i, j): 
    return i + (SN2(p) - SN2(p - j))

def SI3(p, i, j, k):
    return SI2(p - k, i, j) + SN3(p) - SN3(p - k)

def SL1(p):
    for i in range(1, p):
        yield i

def SL2(p):
    for i in range(1, p - 1):
        for j in range(1, p - i):
            yield i, j

def SL3(p):
    for i in range(1, p - 2):
        for j in range(1, p - i):
            for k in range(1, p - i - j):
                yield i, j, k

def GmshLexOrder_SEG(p, node=0):
    index = lambda i: SI1(p, i)

    lex = - np.ones(SN1(p))
    
    if p == 0:
        lex[0] = node; node += 1
        return lex, node
    
    lex[index(0)] = node; node += 1
    lex[index(p)] = node; node += 1
    if p == 1:
        return lex, node
    
    for i in SL1(p):
        lex[index(i)] = node; node += 1
        
    return lex, node

def GmshLexOrder_TRI(p, node=0):
    index = lambda i, j: SI2(p, i, j)
    
    lex = - np.ones(SN2(p))
    
    if p == 0:
        lex[0] = node; node += 1
        return lex, node
    
    lex[index(0, 0)] = node; node += 1
    lex[index(p, 0)] = node; node += 1
    lex[index(0, p)] = node; node += 1
    
    if p == 1:
        return lex, node

    for i in SL1(p):
        lex[index(i, 0)]     = node; node += 1
    for j in SL1(p):
        lex[index(p - j, j)] = node; node += 1
    for j in SL1(p):
        lex[index(0, p - j)] = node; node += 1

    if p == 2:
        return lex, node
    
    sub, node = GmshLexOrder_TRI(p - 3, node);
    for _, (j, i) in enumerate(SL2(p)):
        lex[index(i, j)] = sub[_];

    return lex, node

def GmshLexOrder_TET(p, node=0):
    index = lambda i, j, k: SI3(p, i, j, k)
    lex = - np.ones(SN3(p))
    
    if p == 0:
        lex[0] = node; node += 1
        return lex, node
    lex[index(0, 0, 0)] = node; node += 1
    lex[index(p, 0, 0)] = node; node += 1
    lex[index(0, p, 0)] = node; node += 1
    lex[index(0, 0, p)] = node; node += 1
    
    if p == 1:
        return lex, node
    
    # internal edge nodes 
    for i in SL1(p): lex[index(    i,     0,     0)] = node; node += 1
    for j in SL1(p): lex[index(p - j,     j,     0)] = node; node += 1
    for j in SL1(p): lex[index(    0, p - j,     0)] = node; node += 1
    for k in SL1(p): lex[index(    0,     0, p - k)] = node; node += 1
    for j in SL1(p): lex[index(    0,     j, p - j)] = node; node += 1
    for i in SL1(p): lex[index(    i,     0, p - i)] = node; node += 1
    
    if p == 2:
        return lex, node
    
    # /* internal face nodes */
    sub, node = GmshLexOrder_TRI(p - 3, node)
    for _, (i, j) in enumerate(SL2(p)): 
        lex[index(i, j, 0)] = sub[_]
        
    sub, node = GmshLexOrder_TRI(p - 3, node);
    for _, (k, i) in enumerate(SL2(p)): 
        lex[index(i, 0, k)] = sub[_]
        
    sub, node = GmshLexOrder_TRI(p - 3, node);
    for _, (j, k) in enumerate(SL2(p)): 
        lex[index(0, j, k)] = sub[_]
        
    sub, node = GmshLexOrder_TRI(p - 3, node);
    for _, (j, i) in enumerate(SL2(p)): 
        lex[index(i, j, p - i - j)] = sub[_]
        
    if p == 3: 
        return lex, node

    # internal cell nodes */
    sub, node = GmshLexOrder_TET(p - 4, node);
    for _, (k, j, i) in enumerate(SL3(p)):
        lex[index(i, j, k)] = sub[_];

    return lex, node

