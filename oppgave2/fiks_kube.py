"""
Reparerer en .bip-fil skrevet av ueye-tool på Windows.

Feil: programmet skriver filen i tekstmodus, så Windows setter inn en ekstra
byte 0x0D foran hver byte 0x0A. Alle data etter første innsatte byte blir
forskjøvet, og pikselverdiene blir feil (opptil ~65000 i stedet for maks 4095).

Reparasjon: fjern 0x0D som står rett foran 0x0A (b"\r\n" -> b"\n").
Dette er nøyaktig det motsatte av det Windows gjorde, så dataene blir eksakt
gjenopprettet. Til slutt sjekkes det at filstørrelsen er lines*samples*bands*2.

Skriv inn mappen under, lagre, og kjør:  python fiks_kube.py
Lager <navn>_fikset.bip og <navn>_fikset.hdr i samme mappe.
Krever ledig diskplass tilsvarende én kube (~16 GB).
"""
import os
import glob
import shutil

# ======== ENDRE BARE DENNE LINJEN: mappen der .bip- og .hdr-filen ligger ========
MAPPE = r"C:\opt\capturesf25e35"
# ===============================================================================

bip_filer = [f for f in glob.glob(os.path.join(MAPPE, "*.bip")) if not f.endswith("_fikset.bip")]
if len(bip_filer) != 1:
    raise SystemExit(f"Fant {len(bip_filer)} .bip-filer i {MAPPE}: {bip_filer}")
inn = bip_filer[0]
ut = inn[:-4] + "_fikset.bip"

hdr = {}
for linje in open(inn[:-4] + ".hdr"):
    if "=" in linje:
        k, v = linje.split("=", 1)
        hdr[k.strip()] = v.strip()
forventet = int(hdr["lines"]) * int(hdr["samples"]) * int(hdr["bands"]) * 2

print("Inn :", inn, f"({os.path.getsize(inn) / 1e9:.3f} GB)")
print("Ut  :", ut, f"(forventet {forventet / 1e9:.3f} GB)")

BLOKK = 64 * 1024 * 1024
rest = b""
with open(inn, "rb") as fi, open(ut, "wb") as fo:
    lest = 0
    while True:
        data = fi.read(BLOKK)
        if not data:
            break
        lest += len(data)
        data = rest + data
        # en 0x0D helt til slutt kan høre sammen med 0x0A i neste blokk
        if data.endswith(b"\r"):
            rest, data = b"\r", data[:-1]
        else:
            rest = b""
        fo.write(data.replace(b"\r\n", b"\n"))
        print(f"\r  {lest / 1e9:.2f} GB lest", end="")
    fo.write(rest)
print()

storrelse = os.path.getsize(ut)
shutil.copy(inn[:-4] + ".hdr", ut[:-4] + ".hdr")
if storrelse == forventet:
    print("OK: filstørrelsen stemmer nøyaktig. Bruk", ut)
else:
    print(f"ADVARSEL: {storrelse} byte, forventet {forventet} byte (avvik {storrelse - forventet})")