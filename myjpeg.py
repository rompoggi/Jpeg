# This project's author is Romain Poggi.
# Completed on Monday the 20th of June

from functools import lru_cache
import math


def comRem(line):
    """Removes the comments of a line of text and then strips the string.
    """
    if '#' in line:
        line = line[:line.index('#')]
    return line.strip()


def sliSpace(line):
    """Keeps only the non space elements of split line, i.e :
    '   255    0 255   ' will yield '255','0' and '255'
    """
    l = line.split(' ')
    return [x for x in l if x != '']


def sliMany(line):
    """Combines sliSpace and comRem, first applying the later one.
    """
    return sliSpace(comRem(line))


def ppm_tokenize(stream):
    """generator which yields every element of a stream which is correctly formated: no spaces and rid of comments
    """
    for line in stream:
        line = sliMany(line)
        if len(line) == 0:
            continue
        yield from line


def ppm_load(stream):
    """Stores the content of a ppm file into 3 variables: height of the image, width of the image,
    and a 2D array of pixels made up of tuples of length 3 for their R,G,B values
    """
    g = ppm_tokenize(stream)
    filetype = next(g)
    w, h = int(next(g)), int(next(g))
    maxsize = next(g)
    img = [[0 for _ in range(w)] for _ in range(h)]

    for y in range(h):
        for x in range(w):
            img[y][x] = (int(next(g)), int(next(g)), int(next(g)))
    return w, h, img


def ppm_save(w, h, img, output, maxsize=255):
    """ppm_save saves an image of size w x h with pixels stores in img in an output file
    Args:
        param1 (int): width of the image
        param2 (int): height of the image
        param3 (list): matrix of rgb pixels
        param4 (str): outfile where the image is stored.
    Returns:
        None
    """
    with open(output, 'w') as out:
        out.write('P3\n')
        out.write(f'{w}\n{h}\n')
        out.write(f'{maxsize}\n')
        for y in range(h):
            for x in range(w):
                for z in img[y][x]:
                    out.write(f'{z}\n')


def inrange(inp, min=0, max=255):
    """ In range is a function which sets the value of an input value to a min if its value is below
    the min and to a max if its value is above the max, set by default to 0 and 255 respectively.
    Args:
        param1 (int): Integer to normalize
        param2 (int): Minimum value of input
        param3 (int): Maximum value of input
    Returns:
        int: The return value that is normalized
    """
    if inp < min:
        inp = min
    elif inp > max:
        inp = max
    return inp


def RGB2YCbCr(r, g, b):
    """Connverts a RGB tuple into a YCbCr tuple
    """
    Y = round(0 + 0.299 * r + 0.587 * g + 0.114 * b)
    Cb = round(128 - 0.168736 * r - 0.331264 * g + 0.5 * b)
    Cr = round(128 + 0.5 * r - 0.41868 * g + -0.081312 * b)
    Y, Cb, Cr = inrange(Y, 0, 255), inrange(Cb, 0, 255), inrange(Cr, 0, 255)

    return (Y, Cb, Cr)


def YCbCr2RGB(Y, Cb, Cr):
    """Converts a YCbCr tuple into a RGB tuple
    """
    R = round(Y + 1.402*(Cr-128))
    G = round(Y - 0.344136 * (Cb-128) - 0.714136 * (Cr-128))
    B = round(Y + 1.772 * (Cb-128))
    R, G, B = inrange(R, 0, 255), inrange(G, 0, 255), inrange(B, 0, 255)
    return (R, G, B)


def img_RGB2YCbCr(img):
    """Converts a matrix of RGB pixels into 3 matrices such that a tuple of the entries i,j of each
    matrix is the representation of a pixel of the image in the YCbCr color space"""
    h, w = len(img), len(img[0])
    Y = [[0 for _ in range(w)]*h]
    Cb = [[0 for _ in range(w)]*h]
    Cr = [[0 for _ in range(w)]*h]

    for y in range(h):
        for x in range(w):
            Y[y][x], Cb[y][x], Cr[y][x] = RGB2YCbCr(img[y][x])
    return (Y, Cb, Cr)


def img_YCbCr2RGB(Y, Cb, Cr):
    """Converts 3 matrices containing the Y,Cb and Cr elements of a pixel into a single matrix
       whose entry is the RGB matrix of the image
    """
    h, w = len(Y), len(Y[0])
    img = [[0 for _ in range(w)]*h]

    for y in range(h):
        for x in range(w):
            img[y][x] = YCbCr2RGB(Y[y][x], Cb[y][x], Cr[y][x])

    return img


def avgmat3(M):
    """Computes the average value of a matrix so that the entries with value None do not count
    in the average
    """
    l = []
    for y in range(len(M)):
        for x in range(len(M[0])):
            if M[y][x] is not None:
                l.append(M[y][x])
    return sum(l)/len(l)


def avgmat2(M):
    """Computes the average of all entries of a matrix
    """
    return sum(sum(x) for x in M)/(len(M)*len(M[0]))


def avgmat(M):
    """Computes the average value of a matrix so that the entries with value None do not count
    in the average
    """
    res, i = 0, 0
    for y in range(len(M)):
        for x in range(len(M[0])):
            if M[y][x] is not None:
                res += M[y][x]
                i += 1
    return res/i if i != 0 else 0


def subsampling(w, h, C, b, a):
    """function subsampling(w, h, C, a, b) that performs & returns the subsampling of the channel C
     (of size w x h) in the a:b subsampling mode.
    """
    # Try make it work without the None
    wout, hout = math.ceil(w/b), math.ceil(h/a)
    if w % b != 0:
        for x in C:
            x += [None]*(b-(w % b))
    if h % a != 0:
        C += [[None for _ in range(w)]] * (a-(h % a))
    outmat = [[0 for _ in range(wout)] for _ in range(hout)]
    for y in range(hout):
        for x in range(wout):
            outmat[y][x] = avgmat([[C[y*a+j][x*b+i]
                                    for i in range(b)]for j in range(a)])
    return outmat


A = [
    [1,  2,  3,  4,  5,  6,  7,  8,  9, 10],
    [2,  3,  4,  5,  6,  7,  8,  9, 10,  1],
    [3,  4,  5,  6,  7,  8,  9, 10,  1,  2],
    [4,  5,  6,  7,  8,  9, 10,  1,  2,  3],
    [5,  6,  7,  8,  9, 10,  1,  2,  3,  4],
    [6,  7,  8,  9, 10,  1,  2,  3,  4,  5],
    [7,  8,  9, 10,  1,  2,  3,  4,  5,  6],
    [8,  9, 10,  1,  2,  3,  4,  5,  6,  7],
    [9, 10,  1,  2,  3,  4,  5,  6,  7,  8],
]

print(subsampling(10, 9, A, 4, 4))

# def subsampling2(w, h, C, a, b):
#     # Try make it work without the None
#     wout, hout = math.ceil(w/a), math.ceil(h/b)
#     if w % a != 0:
#         for x in C:
#             x += [x[-1]]*(a-(w % a))
#     if h % b != 0:
#         C += C[-1] * (b-(h % b))
#     outmat = [[0 for _ in range(wout)] for _ in range(hout)]
#     for y in range(hout):
#         for x in range(wout):
#             outmat[y][x] = avgmat2([[C[y*b+j][x*a+i]
#                                    for i in range(a)]for j in range(b)])
#     return outmat


def extrapolate(w, h, C, a, b):
    """function extrapolate(w, h, C, a, b) that does the inverse operation,
     where w & h denotes the size of the channel before the subsampling has been applied.
    """
    out = [[0 for _ in range(w)] for _ in range(h)]

    for y in range(h):
        for x in range(w):
            out[y][x] = C[y//a][x//b]
    return out


# def block_splitting2(w, h, C):
#     """function block_splitting(w, h, C) that takes a channel C and that yield all the 8x8 subblocks
#     of the channel, line by line, from left to right. If the channel data does not represent an integer
#     number of blocks, then we extend the extra entries into a new block repeating the last column and
#     the last row
#     """
#     a, b = 8, 8
#     wout, hout = math.ceil(w/a), math.ceil(h/b)
#     if w % a != 0:
#         for x in C:
#             x += [C[0][-1]]*(a-(w % a))
#     if h % b != 0:
#         C += [C[-1][1] for _ in range(w)] * (b-(h % b))
#     outmat = [[0 for _ in range(wout)] for _ in range(hout)]
#     for y in range(hout):
#         for x in range(wout):
#             outmat[y][x] = [[C[y*b+j][x*a+i]
#                              for i in range(a)]for j in range(b)]
#     return outmat


def block_splitting(w, h, C):
    """function block_splitting(w, h, C) that takes a channel C and that yield all the 8x8 subblocks 
    of the channel, line by line, from left to right. If the channel data does not represent an integer
    number of blocks, then we extend the extra entries into a new block repeating the last column and
    the last row
    """
    a, b = 8, 8
    wout, hout = math.ceil(w/a), math.ceil(h/b)
    if w % a != 0:
        for x in C:
            x += [x[-1]]*(a-(w % a))
    if h % b != 0:
        C += [C[-1]] * (b-(h % b))
    for y in range(hout):
        for x in range(wout):
            yield [[C[y*b+j][x*a+i]for i in range(a)]for j in range(b)]


@ lru_cache
def DCTcoef(n):
    """Computes the DCT coefficients of a n x n matrix. It is stored in a cache.
    """
    C = [[0 for _ in range(n)] for _ in range(n)]
    for j in range(n):
        for i in range(n):
            if j == 0:
                C[j][i] = math.sqrt(1/n)
            else:
                C[j][i] = math.sqrt(2/n)*math.cos(math.pi/n*((i)+1/2)*(j))
    return C


def MatVectMult(M, v):
    """Multiply matrix M by vector v"""
    n = len(v)
    out = [0] * n
    for i in range(n):
        out[i] = round(sum([v[j]*M[i][j] for j in range(n)]), 2)
    return out


def TransposeMat(M):
    """Computes the transpose of a Matrix
    """
    return [[M[i][j]for i in range(len(M))]for j in range(len(M[0]))]


def matProd(A, B):
    """Computes the matrix product of A by B"""
    m = len(A)
    n = len(A[0])
    if len(B) != n:
        return None  # the size does not match
    p = len(B[0])
    C = [None] * m
    for i in range(m):
        C[i] = [0] * p
        for k in range(p):
            for j in range(n):
                C[i][k] += A[i][j] * B[j][k]
    return C


def DCT(v):
    """computes and return the DCT-II of the vector v"""
    n = len(v)
    M = DCTcoef(n)
    return MatVectMult(M, v)


def IDCT(v):
    """computes the inverse DCT-II of the vector v"""
    n = len(v)
    M = TransposeMat(DCTcoef(n))
    return MatVectMult(M, v)


def DCT2(m, n, A):
    """Computes the 2D DCT-II of a matrix A of dimension m rows and n columns"""
    mMat = DCTcoef(m)
    nMat = DCTcoef(n)
    A1 = matProd(mMat, A)
    tnMat = TransposeMat(nMat)
    Abar = matProd(A1, tnMat)
    return Abar


def IDCT2(m, n, A):
    """Computes the 2D DCT-II of a matrix Abar of dimension m rows and n columns"""
    Cm = DCTcoef(m)
    Cn = DCTcoef(n)
    tCm = TransposeMat(Cm)
    A1 = matProd(tCm, A)
    A = matProd(A1, Cn)
    return A


def redalpha(n):
    """takes a non-negative integer i and that returns a pair (s, k) s.t.
        s an integer in the set {-1,1},
        k an integer in the range {0..8}, and
        alpha(i)=s⋅alpha(k)"""
    m = n/16
    m -= 2*(m//2)
    s = 1
    if m > 1:
        s = -1
        m -= 1
    if m > 0.5:
        m = 1-m
        s = s*(-1)
    return (s, int(16*m))


def ncoeff8(i, j):
    """takes two integers i & j in the range {0..8} and that returns a pair (s, k) s.t.
        s an integer in the set {-1,1},
        k an integer in the range {0,8}, and
        C[i,j] = s * alpha(k)"""
    if i == 0:
        return (1, 4)
    if j == 0:
        return (1, i)
    n = (i)*(2*j+1)
    return redalpha(n)


@ lru_cache
def alphacoeff(i):
    """Computes the alpha coefficient of the given input"""
    return math.cos(math.pi*i/16)


# We compute the alpha coefficients of the DCT Chen right now, and they are stores in a cache.
for i in range(8):
    alphacoeff(i)


def DCT_Chen(A, a=1):
    """takes an 8x8 matrix A of numbers (integers and/or floating point numbers) and that returns
    the 2D DCT-II transform of A, rounding at a decimals, 1 by default"""
    for i in range(8):
        A[i] = DCT_ChenAux(A[i], a)
    A = TransposeMat(A)
    for i in range(8):
        A[i] = DCT_ChenAux(A[i], a)
    return TransposeMat(A)


def DCT_ChenAux(v, a=1):
    """Computes the 1D DCT of the vector v with rounding at a decimals, 1 by default"""
    v1 = alphacoeff(4) * sum(v)
    v2 = alphacoeff(1)*(v[0]-v[-1])+alphacoeff(3)*(v[1]-v[-2]) + \
        alphacoeff(5)*(v[2]-v[-3])+alphacoeff(7)*(v[3]-v[-4])
    v3 = alphacoeff(2)*(v[0]-v[3]-v[4]+v[-1]) + \
        alphacoeff(6)*(v[1]-v[2]+v[-2]-v[-3])
    v4 = alphacoeff(3)*(v[0]-v[-1])+alphacoeff(7) * \
        (-v[1]+v[-2])+alphacoeff(1)*(-v[2]+v[-3])+alphacoeff(5)*(-v[3]+v[4])
    v5 = alphacoeff(4)*(v[0]-v[1]-v[2]+v[3]+v[4]-v[5]-v[6]+v[7])
    v6 = alphacoeff(5)*(v[0]-v[-1])+alphacoeff(1)*(-v[1]+v[-2]) + \
        alphacoeff(7)*(v[2]-v[-3])+alphacoeff(3)*(v[3]-v[4])
    v7 = alphacoeff(6)*(v[0]-v[3]-v[4]+v[-1]) + \
        alphacoeff(2)*(-v[1]+v[2]-v[-2]+v[-3])
    v8 = alphacoeff(7)*(v[0]-v[-1])+alphacoeff(5)*(-v[1]+v[-2]) + \
        alphacoeff(3)*(v[2]-v[-3])+alphacoeff(1)*(-v[3]+v[4])
    return [round(v1/2, a), round(v2/2, a), round(v3/2, a), round(v4/2, a), round(v5/2, a), round(v6/2, a), round(v7/2, a), round(v8/2, a)]


def IDCT_Chen(A, a=1):
    """takes an 8x8 matrix A of numbers (integers and/or floating point numbers) and 
    that returns the 2D DCT-II inverse transform of A, with rounding at a decimal, 1 by default"""
    for i in range(8):
        A[i] = IDCT_ChenAux(A[i], a)
    A = TransposeMat(A)
    for i in range(8):
        A[i] = IDCT_ChenAux(A[i], a)
    return TransposeMat(A)


def IDCT_ChenAux(v, a=1):
    """Computes the inverse 1D DCT of the vector v with rounding at a decimals, 1 by default"""
    omg0 = alphacoeff(4)*(v[0]+v[4]) + alphacoeff(2) * \
        (v[2]) + alphacoeff(6)*(v[6])
    omg1 = alphacoeff(4)*(v[0]-v[4]) + alphacoeff(6) * \
        v[2] + alphacoeff(2)*(-v[6])
    omg2 = alphacoeff(4)*(v[0]-v[4]) + alphacoeff(6) * \
        (-v[2]) + alphacoeff(2)*(v[6])
    omg3 = alphacoeff(4)*(v[0]+v[4]) + alphacoeff(2) * \
        (-v[2]) + alphacoeff(6)*(-v[6])
    tht0 = sum(alphacoeff(i)*v[i] for i in range(1, 8, 2))
    tht1 = alphacoeff(3)*v[1]+alphacoeff(7)*(-v[3]) + \
        alphacoeff(1)*(-v[5])+alphacoeff(5)*(-v[7])
    tht2 = alphacoeff(5)*(v[1])+alphacoeff(1)*(-v[3]) + \
        alphacoeff(7)*(v[5])+alphacoeff(3)*v[7]
    tht3 = alphacoeff(7)*v[1]+alphacoeff(5)*(-v[3]) + \
        alphacoeff(3)*(v[5])+alphacoeff(1)*(-v[7])
    v1 = omg0 + tht0
    v2 = omg1 + tht1
    v3 = omg2 + tht2
    v4 = omg3 + tht3
    v5 = omg0 - tht0
    v6 = omg1 - tht1
    v7 = omg2 - tht2
    v8 = omg3 - tht3
    return [round(v1/2, a), round(v2/2, a), round(v3/2, a), round(v4/2, a), round(v8/2, a), round(v7/2, a), round(v6/2, a), round(v5/2, a)]


def quantization(A, Q):
    """takes two 8x8 matrices of numbers (integers and/or floating point numbers) 
    and that returns the quantization of A by Q"""
    n = len(A)
    C = [[0 for _ in range(n)] for _ in range(n)]
    for y in range(n):
        for x in range(n):
            C[y][x] = round(A[y][x]/Q[y][x])
    return C


def quantizationI(A, Q):
    """takes two 8x8 matrices of numbers (integers and/or floating point numbers)
     and that returns the inverse quantization of A by Q"""
    n = len(A)
    C = [[0 for _ in range(n)] for _ in range(n)]
    for y in range(n):
        for x in range(n):
            C[y][x] = round(A[y][x]*Q[y][x])
    return C


def Qmatrix(isY, phi):
    """takes a boolean isY and a quality factor phi. 
    If isY is True, it returns the standard JPEG quantization matrix for the luminance channel, 
    lifted by the quality factor phi. 
    If isY is False, it returns the standard JPEG quantization matrix for the chrominance channel,
    lifted by the quality factor phi."""
    n = 8  # size Quantization matrix
    LQM = [  # Qmatrix for Luminance
        [16, 11, 10, 16,  24,  40,  51,  61],
        [12, 12, 14, 19,  26,  58,  60,  55],
        [14, 13, 16, 24,  40,  57,  69,  56],
        [14, 17, 22, 29,  51,  87,  80,  62],
        [18, 22, 37, 56,  68, 109, 103,  77],
        [24, 35, 55, 64,  81, 104, 113,  92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103,  99],
    ]
    CQM = [  # Qmatrix for chrominance
        [17, 18, 24, 47, 99, 99, 99, 99],
        [18, 21, 26, 66, 99, 99, 99, 99],
        [24, 26, 56, 99, 99, 99, 99, 99],
        [47, 66, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
    ]

    Q = LQM if isY else CQM
    S = 200-phi if phi >= 50 else round(5000/phi)

    C = [[0 for _ in range(n)] for _ in range(n)]
    for y in range(n):
        for x in range(n):
            C[y][x] = math.ceil((50+S*Q[y][x])/100)
    return C


def zigzag(A):
    """takes a 8x8 row-major matrix and that returns a generator that yields all the values of A,
    following the zig-zag ordering.
    """
    n = 8
    dirx, diry = 1, -1
    pos = [0, 0]  # we keep track of the coordinates
    for _ in range(n**2):
        yield A[pos[1]][pos[0]]
        pos[0] += dirx
        pos[1] += diry
        if pos[0] < 0 or pos[1] < 0 or pos[0] >= n or pos[1] >= n:
            # we reach any border, we change direction
            dirx, diry = diry, dirx
        if pos[0] == n:
            # if right border reached, we go bring x back to before, and we bring down by 2 as in when we enter the loop we bring y up by 1 when we need to bring it down by 1.
            pos[0] -= 1
            pos[1] += 2
        if pos[1] == n:  # we reach botton border: bring y back to bottom and x to the right as we went 1 off in the step
            pos[0] += 2
            pos[1] -= 1
        if pos[0] < 0:  # we reach left border, so we bring x back to the right but leave y unchanged since it already went down while reaching the left border
            pos[0] = 0
        if pos[1] < 0:  # we reach top, bring y back to top row, x unchanged since it went right direction alrady
            pos[1] = 0
        #  yield the position we reached at the end of the loop.


def rle0(g):
    """takes a generator that yields integers and that returns a generator that yields
    the pairs obtained from the RLE0 encoding of g"""
    i = 0
    for v in g:
        if v == 0:
            i += 1
            continue
        yield (i, v)
        i = 0
