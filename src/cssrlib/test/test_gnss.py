from cssrlib.gnss import uGNSS, uSIG, uTYP, rSigRnx, sys2char


signals = {}
signals.update({uGNSS.GPS: [uSIG.L1, uSIG.L2, uSIG.L5]})
signals.update({uGNSS.GLO: [uSIG.L1, uSIG.L2, uSIG.L3, uSIG.L4, uSIG.L5]})
signals.update({uGNSS.GAL: [uSIG.L1, uSIG.L5, uSIG.L6, uSIG.L7, uSIG.L8]})
signals.update({uGNSS.BDS: [uSIG.L1, uSIG.L2, uSIG.L5, uSIG.L6, uSIG.L7,
                            uSIG.L8]})
signals.update({uGNSS.QZS: [uSIG.L1, uSIG.L2, uSIG.L5, uSIG.L6]})
signals.update({uGNSS.SBS: [uSIG.L1, uSIG.L5]})
signals.update({uGNSS.IRN: [uSIG.L5, uSIG.L9]})

for sys, sigs in signals.items():
    for sig in sigs:
        rnxSig = rSigRnx(sys, uTYP.C, sig)
        print("{} {} {:9.4f} MHz  {:5.2f} cm".format(
              sys2char(sys), rnxSig.str(),
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

sig = sig.toTyp(uTYP.D).toAtt('A')
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
