#!/usr/bin/python3

#
# 	ec⁴ - The ECC Constant Creator
#
# 	Copyright (C) 2018 Nikolas Rösener
#

import sys, os

# include submodule 'joeecc' as module path
sys.path.append(os.path.abspath("joeecc/"))

import argparse
import re
from math import ceil

from ecc import (
    AffineCurvePoint,
    ECPrivateKey,
    FieldElement,
    MontgomeryCurve,
    ShortWeierstrassCurve,
    TwistedEdwardsCurve,
    getcurvebyname,
)

################################ Program Configuration ################################

PROG_VERSION = "1.0a"

SUPPORTED_CURVES = ["secp256r1", "curve25519", "ed25519", "wei25519", "wei25519.2"]
SUPPORTED_SYNTAX = ["C", "Python", "Integers"]

################################ Argument Parsing ################################

parser = argparse.ArgumentParser(
    description="ec⁴ - The ECC Constant Creator",
    epilog="",
    add_help=True,
    allow_abbrev=True,
)

generator = parser.add_argument_group(
    title="generate", description="Options for generating important constants"
)
syntaxor = parser.add_argument_group(
    title="syntax", description="Options for how constants are displayed syntactically"
)
overwrites = parser.add_argument_group(
    title="syntax overwrites",
    description="Options for overwriting some features of the printed syntax",
)
parser.add_argument("--version", action="version", version=PROG_VERSION)
parser.add_argument(
    "-v",
    "--verbose",
    action="count",
    default=0,
    help="determine the verboseness of the output",
)
generator.add_argument(
    "-c",
    "--curve",
    dest="curves",
    action="append",
    choices=SUPPORTED_CURVES,
    help="determine the curves",
)
syntaxor.add_argument(
    "-t",
    "--target-syntax",
    choices=SUPPORTED_SYNTAX,
    default="C",
    help="Determine the language (syntax)",
)
syntaxor.add_argument(
    "-b",
    "--byte-order",
    choices=["big", "little"],
    default="big",
    help="Determines byte order endianness",
)
syntaxor.add_argument(
    "-w",
    "--word-order",
    choices=["big", "little"],
    default="big",
    help="Determines word order endianness",
)
syntaxor.add_argument(
    "-s",
    "--word-size",
    type=int,
    default=32,
    choices=[1<<exponent for exponent in range(3,9)],
    help="Determines the size of a word in bits",
)
syntaxor.add_argument(
    "-pp",
    "--pretty-print",
    action="store_true",
    default=False,
    help="Prettier multi-line output",
)
generator.add_argument(
    "--with-test-vectors",
    dest="test",
    action="store_true",
    help="Print some test vectors for each curve",
)
parser.add_argument(
    "--test-conversions",
    action="store_true",
    default=False,
    help="Run integrity checks on supported conversions",
)
overwrites.add_argument("--word-delim", default=" ")
overwrites.add_argument("--word-format", default="{}")
overwrites.add_argument("--start-delim", default="")
overwrites.add_argument("--end-delim", default="")
overwrites.add_argument("--hex-case", choices=["LOWER", "UPPER"], default="LOWER")


arg = parser.parse_args()
if not arg.curves:
    arg.curves = SUPPORTED_CURVES


################################ Helper Functions ################################


def separator():
    print("-" * 150)


from random import randrange


def generate_random_point(curve):
    while True:
        x = randrange(0, curve.n)
        points = curve.getpointwithx(x)
        if points:
            (p1, p2) = points
            assert p1.oncurve()
            return p1


def vprint(*args, **kwargs):
    if arg.verbose >= 1:
        print(args, kwargs)


def vvprint(*args, **kwargs):
    if arg.verbose >= 2:
        print(args, kwargs)


def vvvprint(*args, **kwargs):
    if arg.verbose >= 3:
        print(args, kwargs)


vvvprint(parser.parse_args())

################################ Curve Specifications ################################

usedcurve = getcurvebyname("secp256r1")

# General Domain Parameters
p = 57896044618658097711785492504343953926634992332820282019728792003956564819949
n = 7237005577332262213973186563042994240857116359379907606001950938285454250989
h = 8

# Montgommery Domain Parameters
A = 486662

# Edwards Domain Parameters
d = 37095705934669439343138083508754565189542113879843219016388785533085940283555

wei = ShortWeierstrassCurve(
    19298681539552699237261830834781317975544997444273427339909597334573241639236,  # a
    55751746669818908907645289078257140818241103727901012315294400837956729358436,  # b
    p,  # p
    n,  # n (order)
    h,  # h cofactor
    19298681539552699237261830834781317975544997444273427339909597334652188435546,  # G_x
    14781619447589544791020593568409986887264606134616475288964881837755586237401,  # G_y
)
wei._name = "Wei25519"

wei2 = ShortWeierstrassCurve(
    2,  # a
    12102640281269758552371076649779977768474709596484288167752775713178787220689,  # b
    p,  # p
    n,  # n (order)
    h,  # h cofactor
    10770553138368400518417020196796161136792368198326337823149502681097436401658,  # G_x
    5443057586150840565309866898445752861680710333250257752116143977388639873869  # G_y
    # or 846025058902569085893971324802875405096439762854508636287843717141586320219
)
wei2._name = "Wei25519.2"

ed = TwistedEdwardsCurve.TwistedEdwardsCurve(
    (-1),  # a (Edwards)
    d,
    p,
    n,
    h,
    15112221349535400772501151409588531511454012693041857206046113283949847762202,
    46316835694926478169428394003475163141307993866256225615783033603165251855960,
)
ed._name = "Ed25519"

mt = MontgomeryCurve.MontgomeryCurve(
    A,
    1,  # B
    p,
    n,
    h,
    9,  # G_x
    14781619447589544791020593568409986887264606134616475288964881837755586237401,  # G_y
)
mt._name = "Curve25519"

c = (
    51042569399160536130206135233146329284152202253034631822681833788666877215207
)  # sqrt( -(A+2) )
delta = (
    19298681539552699237261830834781317975544997444273427339909597334652188435537
)  # + A // 3

curve = {
    "secp256r1": usedcurve,
    "curve25519": mt,
    "ed25519": ed,
    "wei25519": wei,
    "wei25519.2": wei2,
}

if arg.target_syntax == "C":
    arg.word_delim = ", "
    arg.word_format = "0x{}"
    arg.start_delim = "{"
    arg.end_delim = "};"
    # arg.hex_case = 'UPPER'
    arg.type = "static const uint32_t"
    arg.style = "C"
    arg.comment_symbol = "// "
if arg.target_syntax == "Integers":
    arg.style = "plain"
    arg.comment_symbol = ""
if arg.target_syntax == "Python":
    arg.style = "Python"
    arg.comment_symbol = "# "


################################ Printer Functions ################################


def inttobytes(value, length):
    """Converts a big-endian integer value into a bytes object."""
    return bytes((value >> (8 * i)) & 0xff for i in reversed(range(length)))


def print_int(param, *args, **kwargs):
    return str(int(param))


def print_words(
    param,
    words=int(256 / arg.word_size),
    wordsize=int(arg.word_size / 4),
    byteorder=arg.byte_order,
    wordorder=arg.word_order,
    word_delim=arg.word_delim,
    word_format=arg.word_format,
    start_delim=arg.start_delim,
    end_delim=arg.end_delim,
    hex_case=arg.hex_case,
):
    hexx = (
        param.__int__()
        .to_bytes(length=int((words * wordsize) / 2), byteorder=byteorder)
        .hex()
    )

    if hex_case == "UPPER":
        val = hexx.upper()
    else:
        val = hexx.lower()

    out = start_delim
    for i in range(0, words):
        if arg.pretty_print and i % 8 == 0:
            out = out + "\n\t"
        if wordorder == "big":
            out = out + word_format.format(val[i * wordsize : (i + 1) * wordsize])
        if wordorder == "little":
            out = out + word_format.format(
                val[
                    words * wordsize
                    - (i + 1) * wordsize : words * wordsize
                    - i * wordsize
                ]
            )
        if i != words - 1:
            out = out + word_delim
        else:
            if arg.pretty_print:
                out = out + "\n"

    out = out + end_delim
    return out

syntax = {
    "Integers": print_int,
    "Bytestrings": print_words,
    "C": print_words,
    "C array": print_words,
    "Python": print_int,
}

def print_code(name, data, comment="", style=arg.style, words=int(256 / arg.word_size)):
    if style == "plain":
        print(
            "{name}: {data}{comment}".format(
                name=name, data=data, comment=" (" + comment + ")" if comment else ""
            )
        )
    if style == "Python":
        print(
            "{name} = {data}{comment}".format(
                name=re.sub("[^0-9a-zA-Z_]|^(?=\d)", "_", name.lower()),
                data=data,
                comment=" " + arg.comment_symbol + comment if comment else "",
            )
        )
    if style == "C":
        print(
            "{type} {name}[{words}] = {data} {comment}".format(
                name=name,
                type=arg.type,
                words=words,
                data=data,
                comment=" " + arg.comment_symbol + comment if comment else "",
            )
        )
    if style == "c8":
        print(" {} ; //{}{}".format(data, name, " (" + comment + ")" if comment else ""))


################################ Conversion Functions ################################


def convert(p, target):
    c = (
        51042569399160536130206135233146329284152202253034631822681833788666877215207
    )  # sqrt( -(A+2) )
    delta = (
        19298681539552699237261830834781317975544997444273427339909597334652188435537
    )  # p + A // 3

    if p.curve.curvetype == target.curvetype:
        return p
    if p.curve.curvetype == "shortweierstrass" and target.curvetype == "twistededwards":
        if p.curve.is_neutral(p):
            return target.neutral()
        elif p.x == delta and p.y == 0:
            return AffineCurvePoint(0, -1, target)
        else:
            pa = 3 * p.x - A
            ex = (c * pa) // (3 * p.y)
            ey = (pa - 3) // (pa + 3)
            return AffineCurvePoint(ex.sigint(), ey.sigint(), target)
    if p.curve.curvetype == "shortweierstrass" and target.curvetype == "montgomery":
        if p.curve.is_neutral(p):
            return target.neutral()
        else:
            u = p.x - delta
            v = p.y
            return AffineCurvePoint(u.sigint(), v.sigint(), target)
    if p.curve.curvetype == "montgomery" and target.curvetype == "shortweierstrass":
        if p.curve.is_neutral(p):
            return target.neutral()
        else:
            u = p.x + delta
            v = p.y
            return AffineCurvePoint(u.sigint(), v.sigint(), target)
    if p.curve.curvetype == "montgomery" and target.curvetype == "twistededwards":
        if p.curve.is_neutral(p):
            return target.neutral()
        elif p.x == 0 and p.y == 0:
            return AffineCurvePoint(0, -1, target)
        else:
            u = (c * p.x) // p.y
            v = (p.x - 1) // (p.x + 1)
            return AffineCurvePoint(u.sigint(), v.sigint(), target)
    if p.curve.curvetype == "twistededwards" and target.curvetype == "shortweierstrass":
        if p.curve.is_neutral(p):
            return target.neutral()
        elif p.x == 0 and p.y == -1:
            return AffineCurvePoint(delta, 0, target)
        else:
            wx = (1 + p.y) // (1 - p.y) + delta
            wy = c * (1 + p.y) // ((1 - p.y) * p.x)
            return AffineCurvePoint(wx.sigint(), wy.sigint(), target)
    if p.curve.curvetype == "twistededwards" and target.curvetype == "montgomery":
        if p.curve.is_neutral(p):
            return target.neutral()
        elif p.x == 0 and p.y == -1:
            return AffineCurvePoint(0, 0, target)
        else:
            mx = (1 + p.y) // (1 - p.y)
            my = (c * (1 + p.y)) // ((1 - p.y) * p.x)
            return AffineCurvePoint(mx.sigint(), my.sigint(), target)
    else:
        print(
            "Error! Unknown conversion from %s to %s"
            % (p.curve.curvetype, target.curvetype)
        )


def calculate_barrett_constant(quotient, modulus=(2 ** 32), wordcount=8):
    from decimal import getcontext, setcontext, Decimal

    ctx = getcontext()
    ctx.prec = 128
    setcontext(ctx)
    return ctx.divide_int(Decimal((modulus ** (2 * wordcount))), Decimal(quotient))


def test(p, target):
    return convert(p, target).oncurve()


def test_print(p, target):
    if not test(p, target):
        print("%s not on curve %s!" % (p, target))
        exit(1)


################################ Conversion Integrity Checks ################################

if arg.test_conversions:
    vprint("Running Conversion Tests...")

    for i in range(0, 100):
        pw = generate_random_point(wei)
        test_print(pw, mt)  # wei -> mt
        test_print(pw, ed)  # wei -> ed
        test_print(convert(pw, ed), wei)  # ed -> wei
        test_print(convert(pw, ed), mt)  # ed -> mt
        test_print(convert(pw, mt), wei)  # mt -> wei
        test_print(convert(pw, mt), ed)  # mt -> ed

    test_print(wei.neutral(), mt)  # wei -> mt
    test_print(wei.neutral(), ed)  # wei -> ed
    test_print(ed.neutral(), wei)  # ed -> wei
    test_print(ed.neutral(), mt)  # ed -> mt
    test_print(mt.neutral(), wei)  # mt -> wei
    test_print(mt.neutral(), ed)  # mt -> ed

    test_print(wei.G, mt)  # wei -> mt
    test_print(wei.G, ed)  # wei -> ed
    test_print(ed.G, wei)  # ed -> wei
    test_print(ed.G, mt)  # ed -> mt
    test_print(mt.G, wei)  # mt -> wei
    test_print(mt.G, ed)  # mt -> ed

    assert convert(ed.G, wei) == wei.G
    assert convert(wei.G, ed) == ed.G
    assert convert(ed.G, mt) == mt.G
    assert convert(wei.G, mt) == mt.G
    assert convert(mt.G, wei) == wei.G
    assert convert(mt.G, ed) == ed.G

    delta = 19298681539552699237261830834781317975544997444273427339909597334652188435537
    test_print(AffineCurvePoint(delta, 0, wei), mt)  # wei -> mt
    test_print(AffineCurvePoint(delta, 0, wei), ed)  # wei -> ed
    test_print(AffineCurvePoint(0, -1, ed), wei)  # ed -> wei
    test_print(AffineCurvePoint(0, -1, ed), mt)  # ed -> mt
    test_print(AffineCurvePoint(0, 0, mt), wei)  # mt -> wei
    test_print(AffineCurvePoint(0, 0, mt), ed)  # mt -> ed

    vprint("Conversion Tests Successful")

    separator()

################################ Param Printer ################################


def print_domainparams(curve, f):
    print(arg.comment_symbol + str(curve))
    params = curve.domainparams

    print(arg.comment_symbol + "Relevant domain parameters:")
    if curve.curvetype == "shortweierstrass":
        print_code("a", f(params.a), comment="curve parameter a_4 = a")
        print_code("minus_a", f(-params.a), comment="-a mod p")
        print_code("b", f(params.b), comment="curve parameter a_6 = b")
    if curve.curvetype == "montgomery":
        print_code("A", f(params.a))
        print_code("B", f(params.b))
        print_code("2A", f(params.a * 2), comment="used in Okeya and Sakurai y-coord recovery")
        print_code("2B", f(params.b * 2), comment="used in Okeya and Sakurai y-coord recovery")
    if curve.curvetype == "twistededwards":
        print_code("a", f(params.a), comment="curve twist")
        print_code("d", f(params.d))

    print_code("p", f(params.p), comment="the modulus")
    print_code("n", f(params.n), comment="the group order")
    if curve.curvetype == "shortweierstrass":
        print_code("h", f(params.h), comment="the cofactor")
    print_code("G_x", f(params.G.x), comment="the x coordinate of the base point")
    print_code("G_y", f(params.G.y), comment="the y coordinate of the base point")

    print(arg.comment_symbol + "Prominent field elements:")
    print_code("zero", f(FieldElement(0, params.p)))
    print_code("one", f(FieldElement(1, params.p)))
    print_code("minus_one", f(FieldElement(-1, params.p)))

    print(arg.comment_symbol + "Constants for group operations:")
    print_code(
        "p_r",
        f(params.p ^ 0xfffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffe),
        comment="2^256 - p",
    )
    print_code(
        "pr_squared",
        f(
            FieldElement(
                params.p
                ^ 0xfffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffe,
                params.p,
            ).sqr()
        ),
        comment="(2^256 - p)^2",
    )
    print_code(
        "n_r",
        f(params.n ^ 0xfffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffe),
        comment="order-n xor 2^(bits)-1",
    )
    print_code(
        "mu_n",
        f(calculate_barrett_constant(params.n), words=int((256 + arg.word_size) / arg.word_size)),
        words=int(288 / arg.word_size),
        comment="constant μ for reducing n with Barett modular reduction",
    )
    print_code(
        "mu_p",
        f(calculate_barrett_constant(params.p), words=int((256 + arg.word_size) / arg.word_size)),
        words=int(288 / arg.word_size),
        comment="constant μ for reducing p with Barett modular reduction",
    )
    print_code(
        "mp_inv",
        f((-FieldElement(params.p, 2 ** 16)).inverse(), words=int( ceil(16 / arg.word_size))),
        comment="inverse of -(p mod 2^16)",
    )

    print(arg.comment_symbol + "Constants for conversions:")
    print_code("delta", f(delta), comment="(p+A)/3")
    print_code("c", f(c), comment="sqrt(-(A+2)")
    print_code(
        "c_inv", f(FieldElement(c, wei.domainparams.p).inverse()), comment="inverse of c"
    )

    if arg.test:
        print(arg.comment_symbol + "Random test vectors:")
        if curve.curvetype == "shortweierstrass":
            p = curve.getpointwithx(
                0xde2444bebc8d36e682edd27e0f271508617519b3221a8fa0b77cab3989da97c9
            )
            if not p:
                p = curve.getpointwithx(
                    0xee2444bebc8d36e682edd27e0f271508617519b3221a8fa0b77cab3989da97c9
                )
            if p:
                s1, s2 = p
            else:
                s1 = generate_random_point(curve)
            print_code("Sx", f(s1.x))
            print_code("Sy", f(s1.y))
            p = curve.getpointwithx(
                0x55a8b00f8da1d44e62f6b3b25316212e39540dc861c89575bb8cf92e35e0986b
            )
            if not p:
                p = curve.getpointwithx(
                    0x45a8b00f8da1d44e62f6b3b25316212e39540dc861c89575bb8cf92e35e0986b
                )
            if p:
                t1, t2 = p
            else:
                t1 = generate_random_point(curve)
            print_code("Tx", f(t1.x))
            print_code("Ty", f(t1.y))
            secret = 0xc51e4753afdec1e6b6c6a5b992f43f8dd0c7a8933072708b6522468b2ffb06fd
            print_code("Sec", f(secret))
            a1 = curve.point_addition(s1, t1)
            assert a1.oncurve()
            print_code("AddX", f(a1.x))
            print_code("AddY", f(a1.y))
            m1 = secret * s1
            assert m1.oncurve()
            print_code("MulX", f(m1.x))
            print_code("MulY", f(m1.y))
            d1 = s1 + s1
            assert d1.oncurve()
            print_code("DubX", f(d1.x))
            print_code("DubY", f(d1.y))
            d2 = t1 + t1
            assert d2.oncurve()
            print_code("DubTX", f(d2.x))
            print_code("DubTY", f(d2.y))
            print(arg.comment_symbol + "Constants for testing field operations")
            print_code(
                "Full",
                f(
                    FieldElement(
                        0xffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff,
                        params.p,
                    )
                ),
            )
            print_code("One", f(FieldElement(0x01, params.p)))
            print_code(
                "resultFullAdd",
                f(
                    FieldElement(
                        0xffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff,
                        params.p,
                    )
                    + FieldElement(0x01, params.p)
                ),
            )
            print_code(
                "primeMinusOne",
                f(FieldElement(params.p, params.p) - FieldElement(0x01, params.p)),
            )
            print_code("inv", f(FieldElement(2, params.p).inverse()))
            print_code("one", f(FieldElement(2, params.p) // FieldElement(2, params.p)))
            print_code(
                "resultDoubleMod",
                f(2 * (FieldElement(params.p, params.p) - FieldElement(0x01, params.p))),
            )
            print_code(
                "resultQuadMod",
                f(
                    FieldElement(
                        (
                            FieldElement(params.p, params.p)
                            - FieldElement(0x01, params.p)
                        ).__int__(),
                        params.p ** 2,
                    )
                    ** 2,
                    words=int(512/arg.word_size),
                ),
                words=int(512/arg.word_size),
            )
            print_code(
                "resultFullMod",
                f(FieldElement(params.p, params.p) * FieldElement(params.p, params.p)),
            )
            print_code(
                "orderMinusOne",
                f(FieldElement(params.n, params.p) - FieldElement(0x01, params.p)),
            )
            print_code(
                "orderResultDoubleMod",
                f(
                    2
                    * FieldElement(
                        (
                            FieldElement(params.n, params.p)
                            - FieldElement(0x01, params.p)
                        ).sigint(),
                        params.n,
                    )
                ),
            )
        ecdsaTestMessage = (
            0x48616C6C6F2C205468697320697320612068617368206F662061207365637572
        )
        if ecdsaTestMessage >= params.n:
            ecdsaTestMessage = (
                0x08616C6C6F2C205468697320697320612068617368206F662061207365637572
            )
        ecdsaTestSecret = (
            0x41C1CB6B51247A144321435B7A80E714896A33BBAD7294CA401455A194A949FA
        )
        if ecdsaTestSecret >= params.n:
            ecdsaTestSecret = (
                0x01C1CB6B51247A144321435B7A80E714896A33BBAD7294CA401455A194A949FA
            )
        ecdsaTestRand1 = (
            0x0102030405060708090A0B0C0D0E0F101112131415161718191A1B1C1D1E1F20
        )
        ecdsaTestRand2 = (
            0x01FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF
        )
        msg = ecdsaTestMessage.to_bytes(32, byteorder="big")
        print_code("ecdsaTestMessage", f(ecdsaTestMessage))
        print_code("ecdsaTestSecret", f(ecdsaTestSecret))
        assert ecdsaTestSecret < params.n
        assert ecdsaTestMessage < params.n
        assert ecdsaTestRand1 < params.n
        assert ecdsaTestRand2 < params.n
        assert ecdsaTestSecret > 0
        assert ecdsaTestMessage > 0
        assert ecdsaTestRand1 > 0
        assert ecdsaTestRand2 > 0
        p1 = ecdsaTestSecret * params.G
        print_code("p1x", f(p1.x))
        print_code("p1y", f(p1.y))
        pk = ECPrivateKey(ecdsaTestSecret, curve)
        assert pk.scalar < params.n - 1
        assert pk.scalar > 0
        res1 = pk.ecdsa_sign_hash(msg, k=ecdsaTestRand1)
        print_code("ecdsaTestresultR1", f(res1.r))
        print_code("ecdsaTestresultS1", f(res1.s))
        res2 = pk.ecdsa_sign_hash(msg, k=ecdsaTestRand2)
        print_code("ecdsaTestresultR2", f(res2.r))
        print_code("ecdsaTestresultS2", f(res2.s))
        verify_original = pk.pubkey.ecdsa_verify_hash(msg, res1)
        verify_modified = pk.pubkey.ecdsa_verify_hash(msg, res2)
        assert verify_original
        assert verify_modified


################################ Output ################################

for cname in arg.curves:
    print_domainparams(curve[cname], syntax[arg.target_syntax])
    print()
    print()
    print()
