import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import fuzzy_cabai as kb

try:                                   # scikit-fuzzy bersifat opsional
    import skfuzzy as fuzz
except ImportError:
    fuzz = None

z = kb.UNIVERSE["harga"]


def rp(x):
    return "Rp " + f"{x:,.0f}".replace(",", ".")


# ==========================================================
# 1. DEFUZZIFIKASI -- PYTHON MURNI (NumPy)
# ==========================================================
def defuzz_centroid(x, mu):
    return float(np.sum(x * mu) / np.sum(mu))


def defuzz_bisector(x, mu):
    kumulatif = np.cumsum(mu)
    return float(x[np.searchsorted(kumulatif, kumulatif[-1] / 2.0)])


def _daerah_maksimum(x, mu):
    return x[np.isclose(mu, mu.max(), atol=1e-9)]


def defuzz_mom(x, mu):
    return float(np.mean(_daerah_maksimum(x, mu)))


def defuzz_som(x, mu):
    return float(np.min(_daerah_maksimum(x, mu)))


def defuzz_lom(x, mu):
    return float(np.max(_daerah_maksimum(x, mu)))


METODE = {"centroid": defuzz_centroid, "bisector": defuzz_bisector,
          "mom": defuzz_mom, "som": defuzz_som, "lom": defuzz_lom}


def defuzz_semua(mu):
    return {m: f(z, mu) for m, f in METODE.items()}


# ==========================================================
# 2. MAMDANI (MAX-MIN)
# ==========================================================
def agregasi_mamdani(alpha_label):
    """mu_agg(z) = max_i [ min(alpha_i, mu_Ci(z)) ]"""
    implikasi = {l: np.fmin(a, kb.mf_array("harga", l)) for l, a in alpha_label.items()}
    return np.max(np.vstack(list(implikasi.values())), axis=0), implikasi


def alpha_dari_input(nilai):
    """Nilai crisp -> alpha tiap label output (aturan min, konsekuen sama digabung max)."""
    d = kb.fuzzifikasi_semua(nilai)
    alpha = {"Murah": 0.0, "Sedang": 0.0, "Mahal": 0.0}
    for _, ant, kons in kb.daftar_aturan():
        a = min(d[v][l] for v, l in ant.items())
        alpha[kons] = max(alpha[kons], a)
    return alpha


# ==========================================================
# 3. PERBANDINGAN DENGAN scikit-fuzzy (jika tersedia)
# ==========================================================
def defuzz_skfuzzy(mu):
    if fuzz is None:
        return None
    return {m: float(fuzz.defuzz(z, mu, m)) for m in METODE}


def tampilkan(judul, alpha, mu):
    hasil = defuzz_semua(mu)
    sk = defuzz_skfuzzy(mu)
    print("\n" + "=" * 78)
    print(judul)
    print("alpha:", {k: round(v, 3) for k, v in alpha.items()})
    print("=" * 78)
    print(f"{'Metode':<10}{'Harga (NumPy)':>18}" + (f"{'Harga (skfuzzy)':>18}" if sk else ""))
    for m, v in hasil.items():
        print(f"{m.upper():<10}{rp(v):>18}" + (f"{rp(sk[m]):>18}" if sk else ""))
    if sk is None:
        print("(scikit-fuzzy tidak terpasang -> hanya hasil NumPy ditampilkan)")
    return hasil


def gambar(nama_file, judul, mu, hasil, implikasi=None):
    warna = {"centroid": "red", "bisector": "purple", "mom": "black", "som": "green", "lom": "orange"}
    gaya = {"centroid": "-", "bisector": "--", "mom": "-.", "som": ":", "lom": (0, (5, 1))}
    fig, ax = plt.subplots(figsize=(11, 5.8))
    if implikasi:
        for l, c in zip(implikasi, ("tab:green", "tab:blue", "tab:red")):
            ax.plot(z, kb.mf_array("harga", l), "--", color=c, alpha=0.35, label=f"MF {l}")
    ax.plot(z, mu, "b", lw=2.5, label="Kurva agregasi")
    ax.fill_between(z, 0, mu, color="skyblue", alpha=0.4)
    for m, v in hasil.items():
        ax.axvline(v, color=warna[m], ls=gaya[m], lw=2, label=f"{m.upper()} ({rp(v)})")
    ax.set_title(judul, fontweight="bold")
    ax.set_xlabel("Harga jual cabai (Rp/kg)")
    ax.set_ylabel("Derajat keanggotaan μ(z)")
    ax.set_ylim(-0.05, 1.08)
    ax.grid(True, ls=":", alpha=0.6)
    ax.legend(loc="upper left", fontsize=8.5, ncol=2)
    fig.tight_layout()
    fig.savefig(nama_file, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    # ---- Skenario 1 (tugas): firing strength simulasi ----------------------
    alpha1 = {"Murah": 0.5, "Sedang": 0.8, "Mahal": 0.2}
    mu1, imp1 = agregasi_mamdani(alpha1)
    h1 = tampilkan("SKENARIO 1 -- alpha simulasi (Murah 0,5 / Sedang 0,8 / Mahal 0,2)", alpha1, mu1)
    gambar("komparasi_metode_defuzzifikasi.png",
           "Perbandingan 5 Metode Defuzzifikasi Mamdani -- Skenario 1", mu1, h1, imp1)

    # ---- Skenario 2: alpha dari input nyata via 27 aturan ------------------
    in2 = {"panjang": 8.3, "kesegaran": 1.2, "cacat": 9.0}
    alpha2 = alpha_dari_input(in2)
    mu2, imp2 = agregasi_mamdani(alpha2)
    h2 = tampilkan(f"SKENARIO 2 -- input nyata {in2}", alpha2, mu2)
    gambar("komparasi_defuzz_skenario2.png",
           "Defuzzifikasi pada Input Nyata (8,3 cm; 1,2 hari; 9% cacat)", mu2, h2, imp2)

    # ---- Skenario 3: kurva bimodal (soal analisis 3) ------------------------
    alpha3 = {"Murah": 0.8, "Sedang": 0.0, "Mahal": 0.8}
    mu3, imp3 = agregasi_mamdani(alpha3)
    h3 = tampilkan("SKENARIO 3 -- BIMODAL (Murah 0,8 / Mahal 0,8)", alpha3, mu3)
    print(f"mu agregasi di titik MOM = {mu3[np.abs(z - h3['mom']).argmin()]:.3f} "
          f"(jauh di bawah puncak 0,8: MOM jatuh di lembah, bukan di salah satu puncak)")
    gambar("komparasi_defuzz_bimodal.png", "Kurva Bimodal: Kelemahan MOM", mu3, h3, imp3)

    # ---- Tabel perbandingan ringkas ---------------------------------------
    print("\n" + "=" * 78)
    print("TABEL PERBANDINGAN (Rp/kg)")
    print("=" * 78)
    print(f"{'Metode':<10}{'Skenario 1':>16}{'Skenario 2':>16}{'Skenario 3':>16}")
    for m in METODE:
        print(f"{m.upper():<10}{rp(h1[m]):>16}{rp(h2[m]):>16}{rp(h3[m]):>16}")

    # ---- Beban komputasi (soal analisis 1): hitungan operasi per defuzzifikasi ----
    n = len(z)
    print("\n" + "=" * 78)
    print(f"ESTIMASI OPERASI ARITMATIKA PER DEFUZZIFIKASI (N = {n} titik semesta)")
    print("=" * 78)
    print(f"Centroid : {n} perkalian + {2 * (n - 1)} penjumlahan + 1 pembagian (float, seluruh kurva)")
    print(f"Bisector : {n} penjumlahan kumulatif + pencarian titik setengah luas")
    print(f"MOM/SOM/LOM : {n} pembandingan untuk mencari mu maksimum + rata-rata/min/maks titik puncak")
    print("Pada MCU tanpa FPU, pembagian & perkalian float pada N titik jauh lebih mahal daripada")
    print("pembandingan; MOM/SOM/LOM juga dapat dihitung langsung dari alpha tanpa menelusuri kurva.")
