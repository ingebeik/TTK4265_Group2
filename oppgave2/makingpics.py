"""
Lager et lite utsnitt (~5-15 MB) av en stor datakube, til kontroll av analysen.
Skriv inn mappen under, lagre, og kjør:  python lag_utsnitt.py
Send filen  utsnitt.npz  som kommer ut (havner i samme mappe som dette skriptet).
"""
import os
import glob
import numpy as np

# ======== ENDRE BARE DENNE LINJEN: mappen der .bip- og .hdr-filen ligger ========
MAPPE = r"C:\opt\capturesf10e95"
# ===============================================================================

bip_filer = glob.glob(os.path.join(MAPPE, "*.bip"))
if len(bip_filer) != 1:
    raise SystemExit(f"Fant {len(bip_filer)} .bip-filer i {MAPPE}: {bip_filer}")
bip = bip_filer[0]
print("Bruker:", bip)

hdr = {}
for linje in open(bip[:-4] + ".hdr"):
    if "=" in linje:
        k, v = linje.split("=", 1)
        hdr[k.strip()] = v.strip()
lines, samples, bands = int(hdr["lines"]), int(hdr["samples"]), int(hdr["bands"])
kube = np.memmap(bip, dtype=np.uint16, mode="r", shape=(lines, samples, bands))

# 1) Én hel råramme midt i opptaket: viser hvor spalten og spekteret ligger på sensoren
ramme = np.array(kube[lines // 2])                      # (samples, bands)

# 2) Grov oversikt over hele scenen: hver 4. linje, hver 2. sample, 8 brede bånd
bandgrenser = np.linspace(0, bands, 9).astype(int)
oversikt = np.empty((len(range(0, lines, 4)), samples // 2, 8), np.float32)
for j, l in enumerate(range(0, lines, 4)):
    rad = np.asarray(kube[l, ::2, :], dtype=np.float32)
    for k in range(8):
        oversikt[j, :, k] = rad[:, bandgrenser[k]:bandgrenser[k + 1]].mean(axis=1)
    if j % 100 == 0:
        print(f"\r{l}/{lines} linjer", end="")
print()

ut = os.path.join(os.path.dirname(os.path.abspath(__file__)), "utsnitt.npz")
np.savez_compressed(ut, ramme=ramme, oversikt=oversikt.astype(np.float16),
                    bandgrenser=bandgrenser, shape=np.array([lines, samples, bands]),
                    filnavn=os.path.basename(bip))
print("Lagret", ut)