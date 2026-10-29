"""Felles funksjoner for oppgave 2–5 (innlesing, maskering, beskjæring og transekt)."""
import h5py
import numpy as np
from matplotlib.path import Path
from pyproj import Transformer
from scipy.spatial import cKDTree

# Filnavn
FIL_H2 = "HYPSO2_HSI_aeronetvenice_2025-05-14T10-45-06Z-l1d_L2_OCSMART.h5"
FIL_S3 = "S3B_OL_1_EFR____20250514T091545_20250514T091845_20250515T101142_0179_106_264_2160_ESA_O_NT_004_L2_OCSMART.h5"

# Transekt fra turbid kystvann (Po-plumen) og ut mot klarere vann i sørøst: (lat, lon)
TRANSEKT_START = (44.60, 12.35)
TRANSEKT_SLUTT = (44.45, 13.45)

# Fra lat/lon til UTM 33N (meter), brukes til avstander
_til_utm = Transformer.from_crs("EPSG:4326", "EPSG:32633", always_xy=True)


def les_produkt(filnavn, variabel):
    """Leser lat, lon, et produkt (f.eks. chlor_a(oci)) og L2-flaggene fra en OC-SMART-fil."""
    with h5py.File(filnavn, "r") as f:
        lat = f["Latitude"][:]
        lon = f["Longitude"][:]
        data = f[variabel][:]
        flagg = f["L2_flags"][:]
    return lat, lon, data, flagg


def masker(data, flagg, kun_flagg_0=True, maks=100):
    """Setter ugyldige piksler til NaN."""
    data = data.copy()
    ugyldig = ~np.isfinite(data) | (data <= 0) | (data > maks)  # NaN, negative og ekstreme verdier
    if kun_flagg_0:
        ugyldig |= flagg != 0                                    # flaggede piksler
    data[ugyldig] = np.nan
    return data


def hypso_omriss(lat_h2, lon_h2):
    """Omrisset av HYPSO-2-bildet (alle fire kantene av lat/lon-gridet) som lon- og lat-vektorer."""
    kant_lon = np.concatenate([lon_h2[0, :], lon_h2[:, -1], lon_h2[-1, ::-1], lon_h2[::-1, 0]])
    kant_lat = np.concatenate([lat_h2[0, :], lat_h2[:, -1], lat_h2[-1, ::-1], lat_h2[::-1, 0]])
    return kant_lon, kant_lat


def beskjaer_til_hypso(lat_s3, lon_s3, data_s3, kant_lon, kant_lat):
    """Beskjærer Sentinel-3 til HYPSO-2-omrisset. Piksler utenfor omrisset blir NaN."""
    # Først grov beskjæring med en boks (raskere), så nøyaktig test mot polygonet
    boks = ((lat_s3 >= kant_lat.min()) & (lat_s3 <= kant_lat.max()) &
            (lon_s3 >= kant_lon.min()) & (lon_s3 <= kant_lon.max()))
    rader = np.where(boks.any(axis=1))[0]
    kol = np.where(boks.any(axis=0))[0]
    utsnitt = (slice(rader.min(), rader.max() + 1), slice(kol.min(), kol.max() + 1))
    lat_c, lon_c, data_c = lat_s3[utsnitt], lon_s3[utsnitt], data_s3[utsnitt].copy()

    inni = Path(np.column_stack([kant_lon, kant_lat])).contains_points(
        np.column_stack([lon_c.ravel(), lat_c.ravel()]))
    data_c[~inni.reshape(data_c.shape)] = np.nan
    return lat_c, lon_c, data_c


def transekt_punkter(start=TRANSEKT_START, slutt=TRANSEKT_SLUTT, steg_m=150):
    """Jevnt fordelte punkter langs transektet. Returnerer lat, lon og avstand fra start [km]."""
    x0, y0 = _til_utm.transform(start[1], start[0])
    x1, y1 = _til_utm.transform(slutt[1], slutt[0])
    lengde = np.hypot(x1 - x0, y1 - y0)
    n = int(lengde // steg_m) + 1
    t = np.linspace(0, 1, n)
    x, y = x0 + t * (x1 - x0), y0 + t * (y1 - y0)
    lon, lat = _til_utm.transform(x, y, direction="INVERSE")
    return lat, lon, t * lengde / 1000


def hent_langs_transekt(lat, lon, data, t_lat, t_lon, maks_avstand_m):
    """Henter verdien i nærmeste piksel for hvert transektpunkt (nearest neighbour).
    Punkter uten piksel innenfor maks_avstand_m blir NaN."""
    px, py = _til_utm.transform(lon.ravel(), lat.ravel())   # pikselsentre i meter
    tre = cKDTree(np.column_stack([px, py]))
    tx, ty = _til_utm.transform(t_lon, t_lat)               # transektpunkter i meter
    avstand, idx = tre.query(np.column_stack([tx, ty]))
    verdier = data.ravel()[idx]
    verdier[avstand > maks_avstand_m] = np.nan
    return verdier
