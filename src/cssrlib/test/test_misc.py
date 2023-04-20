from cssrlib.gnss import rSigRnx, uGNSS, uTYP, uSIG, sys2char

sig = rSigRnx()
print(sys2char(sig.sys), sig.str())

sig = rSigRnx("EC5Q")
print(sys2char(sig.sys), sig.str())

sig = rSigRnx(uGNSS.GPS, "D1X")
print(sys2char(sig.sys), sig.str())

sig = rSigRnx(uGNSS.IRN, uTYP.S, uSIG.L1X)
print(sys2char(sig.sys), sig.str())

sig = rSigRnx(uGNSS.GPS, "D1X", "what??")
print(sys2char(sig.sys), sig.str())
