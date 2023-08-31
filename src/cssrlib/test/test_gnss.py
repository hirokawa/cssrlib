from cssrlib.gnss import uGNSS, uSIG, uTYP, rSigRnx, sys2char

signals = {}
signals.update({uGNSS.GPS: [uSIG.L1C, uSIG.L2W, uSIG.L5Q]})
signals.update({uGNSS.GLO: [uSIG.L1P, uSIG.L2C, uSIG.L3X, uSIG.L4A, uSIG.L5B]})
signals.update({uGNSS.GAL: [uSIG.L1C, uSIG.L5Q, uSIG.L6C, uSIG.L7Q, uSIG.L8X]})
signals.update({uGNSS.BDS: [uSIG.L1C, uSIG.L2I, uSIG.L5P, uSIG.L6C, uSIG.L7I,
                            uSIG.L8Q]})
signals.update({uGNSS.QZS: [uSIG.L1C, uSIG.L2L, uSIG.L5Q, uSIG.L6S]})
signals.update({uGNSS.SBS: [uSIG.L1C, uSIG.L5Q]})
signals.update({uGNSS.IRN: [uSIG.L5A, uSIG.L9B]})

for sys, sigs in signals.items():
    for sig in sigs:
        rnxSig = rSigRnx(sys, uTYP.C, sig)
        print("{} {} {:9.4f} MHz  {:5.2f} cm"
              .format(sys2char(sys), rnxSig.str(),
                      rnxSig.frequency(k=-6)*1e-6,
                      rnxSig.wavelength(k=-6)*1e2))
    print()

print()

print("String representation and modification functions")
print()

sig = rSigRnx(uGNSS.GPS, 'L1X')
print("<{:s}>".format(sig.__repr__()))

sig = sig.toTyp(uTYP.C)
print("<{:s}>".format(sig.__repr__()))

sig = sig.toAtt()
print("<{:s}>".format(sig.__repr__()))

sig = sig.toTyp(uTYP.D).toAtt('C')
print("<{:s}>".format(sig.__repr__()))

print()

print("Test constructors")
print()

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
