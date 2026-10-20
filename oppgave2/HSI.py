"""
TTK4265 Assignment 2 - analyse av hyperspektral datakube (HSI v4)

Dekker nøyaktig det oppgaven ber om:
  Del 3   : skriver ut opptaksparametrene (vinkel, tid, f, eksponering) til rapporten
  Del 3.1 : 16 bilder med ~10 nm båndbredde over spektralområdet -> mosaikk
            fargebilde (RGB) av 3 av de 16 bildene, ved siden av mobilbildet
  Del 3.2 : uovervåket klassifisering med k-means

Bruk:
  1. Fyll inn INNSTILLINGER under (filsti, bølgelengdekalibrering fra oppgave 1,
     opptaksparametre).
  2. Kjør med  STEG = "preview"  -> se bildet, les av L0/L1 og S0/S1.
  3. Fyll inn L0/L1/S0/S1, sett  STEG = "analyse"  og kjør igjen.

Kjøring 1 leser hele råkuben (13-16 GB) én gang og lagrer de 16 båndene
i en liten fil (*_16band.npy). Senere kjøringer bruker den filen.

Krever: numpy, matplotlib, scikit-learn
"""
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# =============================================================== INNSTILLINGER
MAPPE = r"C:\opt\capturesf25e35"     # kuben som brukes (reparert med fiks_kube.py)
_filer = glob.glob(os.path.join(MAPPE, "*_fikset.bip"))
if len(_filer) != 1:
    raise SystemExit(f"Fant {len(_filer)} *_fikset.bip i {MAPPE}. Kjør fiks_kube.py først.")
BIP = _filer[0]                       # den reparerte datakuben
HDR = BIP[:-4] + ".hdr"               # header (samme navn)
FOTO = r""            # sti til mobilbildet av scenen (tom streng = ikke vis)

STEG = "analyse"      # "preview" for å velge utsnitt, "analyse" for resultatene

# Bølgelengdekalibrering fra oppgave 1:
#   lambda(x) = c0 + c1*x + c2*x^2 + ...   [nm],  x = båndindeks 0..1935
WL_KOEFF = [304.5, 0.3776, -9.114e-6]   # lab 1, likning (7): lambda(p) = -9.114e-6 p^2 + 0.3776 p + 304.5

# Spektralområdet til instrumentet [nm] (der det faktisk er signal)
LAM_MIN, LAM_MAX = 400.0, 700.0      # brukbart område fra lab 1

N_BILDER = 16         # antall bilder i mosaikken (oppgave: ~16)
BW = 10.0             # båndbredde per bilde [nm] (oppgave: ~10 nm)

RGB_NM = (640.0, 550.0, 460.0)       # ønsket R, G, B; nærmeste av de 16 velges

K = 5                 # antall klynger i k-means

# Utsnitt (fylles inn etter preview):
L0, L1 = 170, 3190    # linjer (skanneretning): stillstand før/etter rotasjon fjernet
S0, S1 = 360, 875     # samples (langs spalten): belyst del av spalten
SNU_OPP_NED = True    # preview var opp ned (himmel nederst)
SPEIL = False         # sett True hvis bildet er speilvendt

# Opptaksparametre (til rapporten, del 3)
VINKEL_GRADER = 50.0  # skannvinkel fra vinkelhjulet [deg]
T_ROT = 102.9         # Play Time i appen [s]
FPS = 25.0            # -f
EKSP_MS = 35.0        # -e
DTHETA = 11.0 / 500   # vinkelstørrelse per piksel [deg] (oppgave seksjon 2.3)
# =============================================================================


def les_hdr(sti):
    """Leser lines, samples, bands fra ENVI .hdr."""
    d = {}
    with open(sti) as f:
        for linje in f:
            if "=" in linje:
                k, v = linje.split("=", 1)
                d[k.strip()] = v.strip()
    assert d["interleave"] == "bip" and d["data type"] == "12", "forventer BIP uint16"
    lines, samples, bands = int(d["lines"]), int(d["samples"]), int(d["bands"])
    if os.path.getsize(BIP) != lines * samples * bands * 2:
        raise SystemExit("Filstørrelsen stemmer ikke med .hdr - kuben er skadet. Kjør fiks_kube.py.")
    return lines, samples, bands


def opptaksparametre():
    """Del 3: tallene som skal stå i rapporten."""
    omega = VINKEL_GRADER / T_ROT               # [deg/s]
    f_ideell = omega / DTHETA                   # f = omega / dtheta  (seksjon 2.3)
    print("Opptaksparametre")
    print(f"  skannvinkel          : {VINKEL_GRADER:.1f} deg")
    print(f"  rotasjonstid T_rot   : {T_ROT:.1f} s")
    print(f"  vinkelhastighet w    : {omega:.4f} deg/s")
    print(f"  framerate brukt f    : {FPS:.2f} Hz")
    print(f"  ideell f = w/dtheta  : {f_ideell:.2f} Hz")
    print(f"  vinkel per ramme w/f : {omega / FPS:.4f} deg (mål {DTHETA:.3f} deg)")
    print(f"  eksponeringstid      : {EKSP_MS:.1f} ms (maks 1/f = {1000 / FPS:.1f} ms)")
    print(f"  linjer i rotasjonen  : f*T_rot = {FPS * T_ROT:.0f} (riktig aspekt krever {VINKEL_GRADER / DTHETA:.0f})")
    print(f"  aspektfaktor         : (w/f)/dtheta = {omega / FPS / DTHETA:.2f}  (>1 klemt, <1 strukket i skanneretningen)")


def bolgelengder(n_bands):
    if WL_KOEFF is None:
        raise ValueError("Fyll inn WL_KOEFF (bølgelengdekalibrering fra oppgave 1).")
    x = np.arange(n_bands, dtype=np.float64)
    return sum(c * x**i for i, c in enumerate(WL_KOEFF))


def lag_16_band():
    """
    Leser råkuben én gang og lager N_BILDER bilder.
    Bilde k = gjennomsnitt av alle råbånd med |lambda - lambda_k| <= BW/2.
    Sentrene lambda_k er jevnt fordelt slik at båndene dekker LAM_MIN..LAM_MAX.
    """
    lines, samples, bands = les_hdr(HDR)
    lam = bolgelengder(bands)
    sentre = np.linspace(LAM_MIN + BW / 2, LAM_MAX - BW / 2, N_BILDER)
    idx = []
    for lc in sentre:
        i = np.where(np.abs(lam - lc) <= BW / 2)[0]
        if len(i) == 0:
            raise ValueError(f"Ingen råbånd rundt {lc:.1f} nm - sjekk WL_KOEFF/LAM_MIN/LAM_MAX")
        idx.append((i.min(), i.max() + 1))
        print(f"  {lc:6.1f} nm : råbånd {i.min()}-{i.max()} ({len(i)} stk)")

    # Samme lesing som i oppgaveteksten (seksjon 2.2):
    #   datacube = np.fromfile(path, dtype='uint16'); datacube.shape = (lines, samples, bands)
    # np.memmap gir nøyaktig samme array, men leser fra disk ved behov i stedet for
    # å laste alle ~16 GB inn i minnet på én gang.
    kube = np.memmap(BIP, dtype=np.uint16, mode="r", shape=(lines, samples, bands))
    ut = np.empty((lines, samples, N_BILDER), np.float32)
    BLOKK = 32                                   # linjer per lesing (~150 MB)
    for l in range(0, lines, BLOKK):
        blk = np.asarray(kube[l:l + BLOKK])
        for k, (a, b) in enumerate(idx):
            ut[l:l + len(blk), :, k] = blk[:, :, a:b].mean(axis=2, dtype=np.float32)
        print(f"\r  leser kube: {min(l + BLOKK, lines)}/{lines} linjer", end="")
    print()
    np.savez(BIP[:-4] + "_16band.npz", bilder=ut, sentre=sentre)


def last_16_band():
    fil = BIP[:-4] + "_16band.npz"
    if not os.path.exists(fil):
        lag_16_band()
    d = np.load(fil)
    B = d["bilder"][L0:L1, S0:S1, :]
    B = np.transpose(B, (1, 0, 2))     # rader = langs spalten, kolonner = skanneretning
    if SNU_OPP_NED:
        B = B[::-1]
    if SPEIL:
        B = B[:, ::-1]
    return B, d["sentre"]


def strekk(img):
    """Kontraststrekk 1-99 persentil til [0, 1] (kun for visning)."""
    lo, hi = np.percentile(img, (1, 99))
    return np.clip((img - lo) / (hi - lo), 0, 1)


def preview(B, sentre):
    """Ett bånd uten utsnitt, med akser i linje-/sample-indeks, for å velge L0/L1/S0/S1."""
    k = N_BILDER // 2
    plt.figure(figsize=(14, 6))
    plt.imshow(strekk(B[:, :, k]), cmap="gray", aspect="equal")
    plt.xlabel("linje (skanneretning)  -> velg L0, L1")
    plt.ylabel("sample (langs spalten)  -> velg S0, S1")
    plt.title(f"Preview {sentre[k]:.0f} nm")
    plt.show()


def mosaikk(B, sentre):
    """Del 3.1: 16 bilder, ~10 nm hver, i et 4x4-rutenett."""
    rad = int(np.ceil(N_BILDER / 4))
    fig, ax = plt.subplots(rad, 4, figsize=(16, 2.6 * rad))
    for a in ax.ravel():
        a.axis("off")
    for k, a in enumerate(ax.ravel()[:N_BILDER]):
        a.imshow(strekk(B[:, :, k]), cmap="gray")
        a.set_title(f"{sentre[k]:.0f} nm (±{BW / 2:.0f} nm)")
    fig.suptitle("Spektral mosaikk (hvert bilde kontraststrukket 1–99 %)")
    fig.tight_layout()
    fig.savefig("mosaikk.png", dpi=150)


def rgb(B, sentre):
    """Del 3.1: fargebilde av 3 av de 16 bildene, ved siden av mobilbildet."""
    kanal = [int(np.argmin(np.abs(sentre - w))) for w in RGB_NM]
    img = np.dstack([strekk(B[:, :, k]) for k in kanal])
    tittel = "RGB: R={:.0f} nm, G={:.0f} nm, B={:.0f} nm".format(*sentre[kanal])

    n = 2 if FOTO else 1
    fig, ax = plt.subplots(1, n, figsize=(8 * n, 5), squeeze=False)
    ax[0, 0].imshow(img)
    ax[0, 0].set_title(tittel)
    ax[0, 0].axis("off")
    if FOTO:
        ax[0, 1].imshow(plt.imread(FOTO))
        ax[0, 1].set_title("Mobilbilde")
        ax[0, 1].axis("off")
    fig.tight_layout()
    fig.savefig("rgb.png", dpi=150)
    plt.imsave("rgb_kompositt.png", img)


def kmeans(B, sentre):
    """Del 3.2: k-means på 16-bånds spekteret i hver piksel."""
    X = B.reshape(-1, B.shape[2])
    km = KMeans(n_clusters=K, n_init=10, random_state=0).fit(X)
    etikett = km.labels_.reshape(B.shape[:2])

    fig, ax = plt.subplots(1, 2, figsize=(16, 5), gridspec_kw={"width_ratios": [2, 1]})
    ax[0].imshow(etikett, cmap="tab10", vmin=0, vmax=9, interpolation="nearest")
    ax[0].set_title(f"k-means, K = {K}")
    ax[0].axis("off")
    for k in range(K):
        ax[1].plot(sentre, km.cluster_centers_[k], "o-", color=plt.cm.tab10(k),
                   label=f"klynge {k} ({np.mean(etikett == k) * 100:.0f} %)")
    ax[1].set_xlabel("bølgelengde [nm]")
    ax[1].set_ylabel("råverdi [DN]")
    ax[1].set_title("Klyngesentre (middelspektrum)")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig("kmeans.png", dpi=150)


if __name__ == "__main__":
    opptaksparametre()
    B, sentre = last_16_band()
    if STEG == "preview":
        preview(B, sentre)
    else:
        mosaikk(B, sentre)
        rgb(B, sentre)
        kmeans(B, sentre)
        plt.show()