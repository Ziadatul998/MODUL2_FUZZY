import numpy as np

# ==========================================================
# 1. FUNGSI KEANGGOTAAN DASAR
# ==========================================================
def trapmf(x, params):
    """Trapesium [a, b, c, d]. a == b atau c == d menghasilkan bahu tegak (shoulder)."""
    a, b, c, d = params
    x = np.asarray(x, dtype=float)
    naik = np.clip((x - a) / (b - a), 0.0, 1.0) if b > a else (x >= a).astype(float)
    turun = np.clip((d - x) / (d - c), 0.0, 1.0) if d > c else (x <= d).astype(float)
    return np.minimum(naik, turun)


def trimf(x, params):
    """Segitiga [a, b, c]  ==  trapesium [a, b, b, c]."""
    a, b, c = params
    return trapmf(x, [a, b, b, c])


# ==========================================================
# 2. DATABASE VARIABEL
# ==========================================================
NAMA_VARIABEL = {
    "panjang":   "Ukuran Panjang Cabai",
    "kesegaran": "Tingkat Kesegaran",
    "cacat":     "Persentase Cacat / Bintik Hitam",
    "harga":     "Harga Jual Cabai",
}
SATUAN = {
    "panjang": "cm",
    "kesegaran": "hari setelah panen",
    "cacat": "%",
    "harga": "Rp/kg",
}

# Semesta pembicaraan (universe of discourse)
UNIVERSE = {
    "panjang":   np.linspace(0, 14, 1401),          # 0 - 14 cm
    "kesegaran": np.linspace(0, 7, 701),            # 0 - 7 hari
    "cacat":     np.linspace(0, 50, 501),           # 0 - 50 %
    "harga":     np.arange(15000, 50000 + 1, 25.0),  # Rp 15.000 - Rp 50.000
}

# Parameter trapesium [a, b, c, d]; segitiga ditulis [a, b, b, c].
# Urutan label = urutan dari kualitas terendah ke tertinggi (dipakai uji monoton).
HIMPUNAN = {
    "panjang": {
        "Kecil":  [0, 0, 3, 6],          # 3-6 cm turun, <=3 cm penuh
        "Sedang": [5, 7, 7, 9],          # 5-9 cm, puncak 7 cm
        "Besar":  [8, 12, 14, 14],       # 8-12 cm naik, >=12 cm penuh
    },
    "kesegaran": {
        "Layu":         [2.0, 3.5, 7, 7],       # > 3 hari (mulai naik di 2 hari)
        "Cukup Segar":  [0.5, 1.5, 1.5, 3.0],   # 1 - 2 hari, puncak 1,5 hari
        "Sangat Segar": [0, 0, 0.5, 1.5],       # < 1 hari
    },
    "cacat": {
        "Banyak": [20, 30, 50, 50],      # > 20 %
        "Sedang": [8, 16, 16, 25],       # 8 - 25 %
        "Sedikit": [0, 0, 5, 10],        # 0 - 10 %
    },
    "harga": {
        "Murah":  [15000, 15000, 18000, 25000],   # Rp 15.000 - 25.000
        "Sedang": [22000, 30000, 30000, 38000],   # Rp 22.000 - 38.000
        "Mahal":  [35000, 42000, 50000, 50000],   # Rp 35.000 - 50.000
    },
}

VAR_INPUT = ["panjang", "kesegaran", "cacat"]
VAR_OUTPUT = "harga"

# Skor kualitas tiap label (0 = terburuk) -> dipakai uji konsistensi monoton
SKOR_LABEL = {
    "panjang":   {"Kecil": 0, "Sedang": 1, "Besar": 2},
    "kesegaran": {"Layu": 0, "Cukup Segar": 1, "Sangat Segar": 2},
    "cacat":     {"Banyak": 0, "Sedang": 1, "Sedikit": 2},
    "harga":     {"Murah": 0, "Sedang": 1, "Mahal": 2},
}


def mf_array(var, label):
    """Nilai keanggotaan seluruh semesta variabel `var` untuk himpunan `label`."""
    return trapmf(UNIVERSE[var], HIMPUNAN[var][label])


def fuzzifikasi(var, nilai):
    """Nilai crisp -> {label: derajat keanggotaan}. Nilai di luar semesta dipotong (clip)."""
    u = UNIVERSE[var]
    nilai = float(np.clip(nilai, u[0], u[-1]))
    return {lbl: float(trapmf(nilai, p)) for lbl, p in HIMPUNAN[var].items()}


def fuzzifikasi_semua(nilai_input):
    """{'panjang': 7.5, 'kesegaran': 2.2, 'cacat': 14} -> derajat bersarang untuk semua variabel."""
    return {v: fuzzifikasi(v, nilai_input[v]) for v in VAR_INPUT}


# ==========================================================
# 3. MATRIKS BASIS ATURAN  (3 x 3 x 3 = 27 aturan)
# ==========================================================
# MATRIKS_ATURAN[kesegaran][panjang] = (harga jika cacat Sedikit, Sedang, Banyak)
URUTAN_CACAT = ("Sedikit", "Sedang", "Banyak")
MATRIKS_ATURAN = {
    "Layu": {
        "Kecil":  ("Murah",  "Murah",  "Murah"),
        "Sedang": ("Murah",  "Murah",  "Murah"),
        "Besar":  ("Sedang", "Murah",  "Murah"),
    },
    "Cukup Segar": {
        "Kecil":  ("Sedang", "Murah",  "Murah"),
        "Sedang": ("Sedang", "Sedang", "Murah"),
        "Besar":  ("Mahal",  "Sedang", "Murah"),
    },
    "Sangat Segar": {
        "Kecil":  ("Sedang", "Sedang", "Murah"),
        "Sedang": ("Mahal",  "Sedang", "Sedang"),
        "Besar":  ("Mahal",  "Mahal",  "Sedang"),
    },
}


def daftar_aturan():
    """Ekspansi matriks menjadi list (id, antecedent dict, label harga). Urutan R1..R27."""
    hasil, n = [], 1
    for kes, baris in MATRIKS_ATURAN.items():
        for pjg, konsekuen in baris.items():
            for cct, harga in zip(URUTAN_CACAT, konsekuen):
                hasil.append((f"R{n}", {"panjang": pjg, "kesegaran": kes, "cacat": cct}, harga))
                n += 1
    return hasil


# ==========================================================
# 4. PEMERIKSAAN BASIS ATURAN
# ==========================================================
def periksa_basis_aturan(aturan=None):
    """Kembalikan dict hasil uji: kelengkapan, konflik, dan konsistensi monoton."""
    aturan = aturan or daftar_aturan()
    peta, konflik = {}, []
    for rid, ant, kons in aturan:
        kunci = tuple(ant[v] for v in VAR_INPUT)
        if kunci in peta and peta[kunci] != kons:
            konflik.append((rid, kunci))
        peta[kunci] = kons

    total_seharusnya = 1
    for v in VAR_INPUT:
        total_seharusnya *= len(HIMPUNAN[v])
    lengkap = len(peta) == total_seharusnya

    # Monoton: memperbaiki satu atribut satu langkah tidak boleh menurunkan label harga
    pelanggaran = []
    for kunci, kons in peta.items():
        for i, v in enumerate(VAR_INPUT):
            skor = SKOR_LABEL[v][kunci[i]]
            for lbl, s in SKOR_LABEL[v].items():
                if s == skor + 1:
                    tetangga = kunci[:i] + (lbl,) + kunci[i + 1:]
                    if SKOR_LABEL["harga"][peta[tetangga]] < SKOR_LABEL["harga"][kons]:
                        pelanggaran.append((kunci, tetangga))
    return {"jumlah_aturan": len(aturan), "kombinasi_unik": len(peta),
            "lengkap": lengkap, "konflik": konflik, "pelanggaran_monoton": pelanggaran}


if __name__ == "__main__":
    print(periksa_basis_aturan())
    print(fuzzifikasi_semua({"panjang": 7.5, "kesegaran": 2.2, "cacat": 14}))
