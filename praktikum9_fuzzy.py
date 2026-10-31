import numpy as np

import fuzzy_cabai as kb


def rp(x):
    return "Rp " + f"{x:,.0f}".replace(",", ".")


# ==========================================================
# 1. FIRING STRENGTH SETIAP ATURAN (AND = MIN)
# ==========================================================
ATURAN = kb.daftar_aturan()


def alpha_aturan(nilai):
    d = kb.fuzzifikasi_semua(nilai)
    return [min(d[v][l] for v, l in ant.items()) for _, ant, _ in ATURAN]


# ==========================================================
# 2. SUGENO (TSK)
# ==========================================================
def hitung_sugeno(alphas, z_values):
    """z* = sum(alpha_i * z_i) / sum(alpha_i)"""
    alphas, z_values = np.asarray(alphas, float), np.asarray(z_values, float)
    total = alphas.sum()
    return 0.0 if total == 0 else float((alphas * z_values).sum() / total)


# Orde 0: konstanta tiap label (mendekati inti himpunan output)
KONSTANTA = {"Murah": 18000.0, "Sedang": 30000.0, "Mahal": 45000.0}

# Orde 1: z = basis_label + 500*(panjang-7) - 1000*(hari-1,5) - 150*(cacat-10), dipotong ke [15000, 50000]
BASIS_ORDE1 = {"Murah": 20000.0, "Sedang": 30000.0, "Mahal": 42000.0}
KOEF_ORDE1 = {"panjang": 500.0, "kesegaran": -1000.0, "cacat": -150.0}


def z_orde1(label, nilai):
    z = (BASIS_ORDE1[label] + KOEF_ORDE1["panjang"] * (nilai["panjang"] - 7.0)
         + KOEF_ORDE1["kesegaran"] * (nilai["kesegaran"] - 1.5)
         + KOEF_ORDE1["cacat"] * (nilai["cacat"] - 10.0))
    return float(np.clip(z, 15000, 50000))


def sugeno_orde0(nilai):
    a = alpha_aturan(nilai)
    return hitung_sugeno(a, [KONSTANTA[k] for _, _, k in ATURAN])


def sugeno_orde1(nilai):
    a = alpha_aturan(nilai)
    return hitung_sugeno(a, [z_orde1(k, nilai) for _, _, k in ATURAN])


# ==========================================================
# 3. TSUKAMOTO  (konsekuen harus monoton)
# ==========================================================
def invers_monoton_naik(alpha, a, b):
    return a + alpha * (b - a)


def invers_monoton_turun(alpha, a, b):
    return b - alpha * (b - a)


# Murah : monoton turun  [15000 -> 25000]   (mu=1 di 15000, mu=0 di 25000)
# Sedang: monoton naik   [22000 -> 38000]
# Mahal : monoton naik   [35000 -> 50000]
INVERS = {
    "Murah":  lambda a: invers_monoton_turun(a, 15000.0, 25000.0),
    "Sedang": lambda a: invers_monoton_naik(a, 22000.0, 38000.0),
    "Mahal":  lambda a: invers_monoton_naik(a, 35000.0, 50000.0),
}


def hitung_tsukamoto(alphas, fungsi_invers_list):
    z_values = [f(a) for a, f in zip(alphas, fungsi_invers_list)]
    return hitung_sugeno(alphas, z_values), z_values


def tsukamoto(nilai):
    a = alpha_aturan(nilai)
    return hitung_tsukamoto(a, [INVERS[k] for _, _, k in ATURAN])[0]


# ==========================================================
# 4. MAMDANI CENTROID (pembanding)
# ==========================================================
Z = kb.UNIVERSE["harga"]
MF_OUT = {l: kb.mf_array("harga", l) for l in kb.HIMPUNAN["harga"]}


def mamdani(nilai):
    a = alpha_aturan(nilai)
    alpha_lbl = {l: 0.0 for l in MF_OUT}
    for ai, (_, _, k) in zip(a, ATURAN):
        alpha_lbl[k] = max(alpha_lbl[k], ai)
    agg = np.max([np.fmin(alpha_lbl[l], MF_OUT[l]) for l in MF_OUT], axis=0)
    return float(np.sum(Z * agg) / np.sum(agg))


# ==========================================================
# 5. PROGRAM UTAMA
# ==========================================================
if __name__ == "__main__":
    # ---- 5.1 Contoh hitung manual: tepat 4 aturan aktif -------------------
    contoh = {"panjang": 7.0, "kesegaran": 2.5, "cacat": 9.0}
    d = kb.fuzzifikasi_semua(contoh)
    print("=" * 78)
    print("CONTOH HITUNG MANUAL (4 ATURAN AKTIF):", contoh)
    print("=" * 78)
    for v in kb.VAR_INPUT:
        print(f"Fuzzifikasi {v:<10}:", {l: round(m, 3) for l, m in d[v].items() if m > 0})
    a = alpha_aturan(contoh)
    aktif = [(i, ATURAN[i][0], ATURAN[i][2], a[i]) for i in range(len(a)) if a[i] > 0]
    print(f"\n{'Aturan':<8}{'Konsekuen':<11}{'alpha':>8}{'z Sugeno-0':>14}{'z Tsukamoto':>14}")
    for i, rid, kons, al in aktif:
        print(f"{rid:<8}{kons:<11}{al:>8.3f}{rp(KONSTANTA[kons]):>14}{rp(INVERS[kons](al)):>14}")
    print(f"\nJumlah aturan aktif = {len(aktif)}")
    print(f"Sugeno orde 0 = {rp(sugeno_orde0(contoh))}")
    print(f"Sugeno orde 1 = {rp(sugeno_orde1(contoh))}")
    print(f"Tsukamoto     = {rp(tsukamoto(contoh))}")
    print(f"Mamdani (centroid) = {rp(mamdani(contoh))}")

    # ---- 5.2 Lima variasi input -----------------------------------------
    variasi = {
        "V1 Cabai layu, kecil, banyak cacat": {"panjang": 4.0, "kesegaran": 4.5, "cacat": 30.0},
        "V2 Sedang, mulai layu":              {"panjang": 7.5, "kesegaran": 2.2, "cacat": 14.0},
        "V3 Besar, baru panen, mulus":        {"panjang": 11.5, "kesegaran": 0.3, "cacat": 3.0},
        "V4 Hampir prima, cacat awal":        {"panjang": 8.3, "kesegaran": 1.2, "cacat": 9.0},
        "V5 Segar tapi cacat banyak":         {"panjang": 6.0, "kesegaran": 1.0, "cacat": 22.0},
    }
    print("\n" + "=" * 100)
    print("KOMPARASI 5 VARIASI INPUT")
    print("=" * 100)
    print(f"{'Variasi':<38}{'Mamdani':>12}{'Sugeno-0':>12}{'Sugeno-1':>12}{'Tsukamoto':>12}"
          f"{'|Sug0-Tsu|':>12}")
    baris = []
    for nama, x in variasi.items():
        m, s0, s1, ts = mamdani(x), sugeno_orde0(x), sugeno_orde1(x), tsukamoto(x)
        baris.append((m, s0, s1, ts))
        print(f"{nama:<38}{rp(m):>12}{rp(s0):>12}{rp(s1):>12}{rp(ts):>12}{rp(abs(s0 - ts)):>12}")

    # ---- 5.3 Selisih rata-rata pada 3.000 input acak ----------------------
    rng = np.random.default_rng(42)
    contoh_acak = [{"panjang": rng.uniform(2, 14), "kesegaran": rng.uniform(0, 7),
                    "cacat": rng.uniform(0, 50)} for _ in range(3000)]
    h = np.array([[mamdani(x), sugeno_orde0(x), sugeno_orde1(x), tsukamoto(x)] for x in contoh_acak])
    nama_m = ["Mamdani", "Sugeno-0", "Sugeno-1", "Tsukamoto"]
    print("\n" + "=" * 78)
    print("SELISIH ABSOLUT RATA-RATA TERHADAP MAMDANI (3.000 input acak)")
    print("=" * 78)
    for j in range(1, 4):
        sel = np.abs(h[:, j] - h[:, 0])
        print(f"{nama_m[j]:<10}: rata-rata {rp(sel.mean())}, maksimum {rp(sel.max())}")
    print("Rata-rata keluaran:", {n: rp(h[:, j].mean()) for j, n in enumerate(nama_m)})
    print("Nilai minimum/maksimum keluaran:")
    for j, n in enumerate(nama_m):
        print(f"  {n:<10}: {rp(h[:, j].min())} -- {rp(h[:, j].max())}")
