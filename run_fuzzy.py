import itertools
import random

import numpy as np

import fuzzy_cabai as kb


# ==========================================================
# 1. KELAS STRUKTUR ATURAN FUZZY
# ==========================================================
class FuzzyRule:
    def __init__(self, rule_id, antecedent, consequent, operator="AND"):
        """
        antecedent : {'panjang': 'Besar', 'kesegaran': 'Sangat Segar', 'cacat': 'Sedikit'}
        consequent : {'harga': 'Mahal'}
        """
        self.rule_id = rule_id
        self.antecedent = antecedent
        self.consequent = consequent
        self.operator = operator.upper()

    def hitung_firing_strength(self, derajat_fuzzifikasi, t_norm="min"):
        """alpha-predikat. AND -> min (Zadeh) atau product; OR -> max."""
        nilai = [derajat_fuzzifikasi[var].get(lbl, 0.0) for var, lbl in self.antecedent.items()]
        if not nilai:
            return 0.0
        if self.operator == "AND":
            return float(min(nilai)) if t_norm == "min" else float(np.prod(nilai))
        if self.operator == "OR":
            return float(max(nilai))
        raise ValueError(f"Operator {self.operator} tidak dikenali.")

    def __str__(self):
        syarat = f" {self.operator} ".join(f"{v} is {l}" for v, l in self.antecedent.items())
        hasil = ", ".join(f"{v} is {l}" for v, l in self.consequent.items())
        return f"[{self.rule_id:<3}] IF {syarat} THEN {hasil}"


# ==========================================================
# 2. BASIS ATURAN (RULE BASE) DARI MATRIKS PAKAR
# ==========================================================
rule_base = [FuzzyRule(rid, ant, {"harga": kons}) for rid, ant, kons in kb.daftar_aturan()]


def cetak_matriks_aturan():
    print("=" * 78)
    print("MATRIKS BASIS ATURAN  (baris: Panjang, kolom: Cacat, tiap tabel: Kesegaran)")
    print("=" * 78)
    nomor = 1
    for kes, baris in kb.MATRIKS_ATURAN.items():
        print(f"\nKesegaran = {kes}")
        print(f"{'Panjang \\ Cacat':<18}" + "".join(f"{c:<14}" for c in kb.URUTAN_CACAT))
        for pjg, kons in baris.items():
            sel = []
            for h in kons:
                sel.append(f"R{nomor}:{h}".ljust(14))
                nomor += 1
            print(f"{pjg:<18}" + "".join(sel))


# ==========================================================
# 3. EVALUASI ATURAN
# ==========================================================
def evaluasi_basis_aturan(derajat, judul, t_norm="min", tampil_semua=False):
    print("\n" + "=" * 78)
    print(judul)
    print("=" * 78)
    aktif = []
    for rule in rule_base:
        alpha = rule.hitung_firing_strength(derajat, t_norm)
        if alpha > 0:
            aktif.append((rule.rule_id, rule.consequent["harga"], alpha))
            print(f"{rule} --> AKTIF (alpha = {alpha:.3f})")
        elif tampil_semua:
            print(f"{rule} --> NONAKTIF")
    print(f"\nRingkasan: {len(aktif)} dari {len(rule_base)} aturan terpicu.")
    for rid, lbl, a in aktif:
        print(f" - {rid:<4}: Harga = {lbl:<7} alpha = {a:.3f}")
    return aktif


def agregasi_per_label(aktif):
    """Aturan yang berkonsekuen sama digabung dengan operator MAX (tahap agregasi)."""
    hasil = {"Murah": 0.0, "Sedang": 0.0, "Mahal": 0.0}
    for _, lbl, a in aktif:
        hasil[lbl] = max(hasil[lbl], a)
    return hasil


# ==========================================================
# 4. PROGRAM UTAMA
# ==========================================================
if __name__ == "__main__":
    cetak_matriks_aturan()

    print("\nDaftar 27 aturan:")
    for r in rule_base:
        print(" ", r)

    # --- Uji kelengkapan & konsistensi basis aturan -------------------------
    cek = kb.periksa_basis_aturan()
    print("\nUJI BASIS ATURAN:", cek)

    # --- TUGAS PRAKTIKUM 6: evaluasi input derajat keanggotaan langsung -----
    # (padanan soal 'Suhu & Beban CPU' -> kriteria fisik cabai)
    derajat_tugas = {
        "panjang":   {"Kecil": 0.2, "Sedang": 0.8, "Besar": 0.0},
        "kesegaran": {"Layu": 0.5, "Cukup Segar": 0.5, "Sangat Segar": 0.0},
        "cacat":     {"Sedikit": 0.0, "Sedang": 0.5, "Banyak": 0.5},
    }
    aktif_tugas = evaluasi_basis_aturan(derajat_tugas, "TUGAS 6: EVALUASI ATURAN (operator MIN)")
    print("\nAgregasi per label konsekuen (MAX):", agregasi_per_label(aktif_tugas))

    # Perbandingan operator AND: MIN (Zadeh) vs PRODUCT
    print("\nPerbandingan operator AND pada kasus tugas:")
    print(f"{'Aturan':<6} {'alpha MIN':>10} {'alpha PRODUCT':>15}")
    for r in rule_base:
        a_min = r.hitung_firing_strength(derajat_tugas, "min")
        if a_min > 0:
            print(f"{r.rule_id:<6} {a_min:>10.3f} {r.hitung_firing_strength(derajat_tugas, 'product'):>15.3f}")

    # --- SKENARIO PENGUJIAN (nilai crisp -> fuzzifikasi -> firing strength) --
    skenario = {
        "A - Panen baru, besar, mulus":          {"panjang": 12.0, "kesegaran": 0.3, "cacat": 3.0},
        "B - Campuran, hampir segar, mulai cacat": {"panjang": 8.3, "kesegaran": 1.2, "cacat": 9.0},
        "C - Kecil, sudah layu, banyak cacat":   {"panjang": 4.0, "kesegaran": 5.0, "cacat": 35.0},
    }
    for nama, nilai in skenario.items():
        d = kb.fuzzifikasi_semua(nilai)
        print(f"\n>>> Skenario {nama}: {nilai}")
        for v in kb.VAR_INPUT:
            tampil = {l: round(m, 3) for l, m in d[v].items() if m > 0}
            print(f"    Fuzzifikasi {v:<10}: {tampil}")
        akt = evaluasi_basis_aturan(d, f"SKENARIO {nama}")
        print("    Agregasi per label:", agregasi_per_label(akt))

    # --- ANALISIS 1: uji kelengkapan (completeness) pada seluruh semesta -----
    print("\n" + "=" * 78)
    print("ANALISIS 1: UJI CAKUPAN ATURAN PADA SELURUH SEMESTA INPUT")
    print("=" * 78)
    grid = {v: np.linspace(kb.UNIVERSE[v][0], kb.UNIVERSE[v][-1], 29) for v in kb.VAR_INPUT}
    alpha_total_min, titik_tanpa_aturan = np.inf, 0
    for p, k, c in itertools.product(grid["panjang"], grid["kesegaran"], grid["cacat"]):
        d = kb.fuzzifikasi_semua({"panjang": p, "kesegaran": k, "cacat": c})
        a_max = max(r.hitung_firing_strength(d) for r in rule_base)
        alpha_total_min = min(alpha_total_min, a_max)
        titik_tanpa_aturan += a_max == 0
    print(f"Titik uji: {29**3}; titik tanpa aturan aktif: {titik_tanpa_aturan}; "
          f"alpha maksimum terkecil di seluruh titik: {alpha_total_min:.3f}")

    # --- ANALISIS 3: urutan aturan tidak memengaruhi hasil ------------------
    print("\n" + "=" * 78)
    print("ANALISIS 3: URUTAN ATURAN TIDAK MEMENGARUHI HASIL")
    print("=" * 78)
    d = kb.fuzzifikasi_semua(skenario["B - Campuran, hampir segar, mulai cacat"])
    asli = {r.rule_id: r.hitung_firing_strength(d) for r in rule_base}
    acak = rule_base[:]
    random.Random(7).shuffle(acak)
    hasil_acak = {r.rule_id: r.hitung_firing_strength(d) for r in acak}
    print("Urutan acak     :", [r.rule_id for r in acak][:8], "...")
    print("Hasil identik   :", asli == hasil_acak)
    print("Agregasi (asli) :", agregasi_per_label([(r.rule_id, r.consequent['harga'], asli[r.rule_id]) for r in rule_base]))
    print("Agregasi (acak) :", agregasi_per_label([(r.rule_id, r.consequent['harga'], hasil_acak[r.rule_id]) for r in acak]))
