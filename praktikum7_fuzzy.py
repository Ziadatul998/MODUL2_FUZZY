import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import fuzzy_cabai as kb


# ==========================================================
# 1. KELAS FUZZY INFERENCE SYSTEM (MAMDANI)
# ==========================================================
class FuzzyInferenceSystem:
    def __init__(self, implication="min", defuzz_fallback=None):
        """implication: 'min' (clipping) atau 'prod' (scaling)."""
        assert implication in ("min", "prod")
        self.implication = implication
        self.inputs = {}        # nama -> {"universe": array, "mf": {label: [a,b,c,d]}}
        self.output = None      # (nama, universe, mf_dict)
        self.rules = []         # list (antecedent, consequent, operator)
        self.defuzz_fallback = defuzz_fallback

    # ---- Knowledge base -------------------------------------------------
    def add_input_variable(self, name, universe, mf_dict):
        self.inputs[name] = {"universe": np.asarray(universe, float), "mf": mf_dict}

    def set_output_variable(self, name, universe, mf_dict):
        self.output = (name, np.asarray(universe, float), mf_dict)

    def add_rule(self, rule_tuple):
        """rule_tuple = (antecedent_dict, label_konsekuen[, operator])"""
        ant, kons = rule_tuple[0], rule_tuple[1]
        op = rule_tuple[2] if len(rule_tuple) > 2 else "AND"
        self.rules.append((ant, kons, op.upper()))

    # ---- Tahap 1: fuzzifikasi ------------------------------------------
    def fuzzify(self, input_values):
        derajat = {}
        for nama, v in self.inputs.items():
            x = float(np.clip(input_values[nama], v["universe"][0], v["universe"][-1]))
            derajat[nama] = {lbl: float(kb.trapmf(x, p)) for lbl, p in v["mf"].items()}
        return derajat

    # ---- Tahap 2: evaluasi aturan (firing strength) --------------------
    def evaluate_rules(self, derajat):
        hasil = []
        for i, (ant, kons, op) in enumerate(self.rules, start=1):
            mu = [derajat[var][lbl] for var, lbl in ant.items()]
            alpha = min(mu) if op == "AND" else max(mu)
            hasil.append((f"R{i}", kons, alpha))
        return hasil

    # ---- Tahap 3: implikasi + agregasi ---------------------------------
    def aggregate(self, firing):
        nama, z, mf = self.output
        # aturan berkonsekuen sama: alpha digabung MAX (agregasi level label)
        alpha_label = {lbl: 0.0 for lbl in mf}
        for _, kons, a in firing:
            alpha_label[kons] = max(alpha_label[kons], a)
        implikasi, agg = {}, np.zeros_like(z)
        for lbl, a in alpha_label.items():
            mu = kb.trapmf(z, mf[lbl])
            implikasi[lbl] = np.minimum(a, mu) if self.implication == "min" else a * mu
            agg = np.maximum(agg, implikasi[lbl])
        return alpha_label, implikasi, agg

    # ---- Tahap 4: defuzzifikasi centroid -------------------------------
    def defuzzify(self, mu_agg):
        z = self.output[1]
        total = np.sum(mu_agg)
        if total == 0:                       # fallback: tidak ada aturan yang aktif
            return (self.defuzz_fallback if self.defuzz_fallback is not None
                    else float((z[0] + z[-1]) / 2)), True
        return float(np.sum(z * mu_agg) / total), False

    # ---- Pipeline lengkap ----------------------------------------------
    def compute(self, input_values_dict, return_detail=False):
        derajat = self.fuzzify(input_values_dict)
        firing = self.evaluate_rules(derajat)
        alpha_label, implikasi, agg = self.aggregate(firing)
        crisp, dipakai_fallback = self.defuzzify(agg)
        if not return_detail:
            return crisp
        return {"crisp": crisp, "derajat": derajat, "firing": firing, "alpha_label": alpha_label,
                "implikasi": implikasi, "agregasi": agg, "fallback": dipakai_fallback}


def bangun_fis(implication="min", subset_aturan=None):
    """Susun FIS cabai dari basis pengetahuan fuzzy_cabai.py."""
    fis = FuzzyInferenceSystem(implication)
    for v in kb.VAR_INPUT:
        fis.add_input_variable(v, kb.UNIVERSE[v], kb.HIMPUNAN[v])
    fis.set_output_variable(kb.VAR_OUTPUT, kb.UNIVERSE["harga"], kb.HIMPUNAN["harga"])
    for rid, ant, kons in (subset_aturan or kb.daftar_aturan()):
        fis.add_rule((ant, kons))
    return fis


# ==========================================================
# 2. TIGA KASUS UJI
# ==========================================================
KASUS = {
    "Kasus 1 (kualitas rendah)": {"panjang": 4.0, "kesegaran": 4.5, "cacat": 30.0},
    "Kasus 2 (kualitas sedang)": {"panjang": 7.5, "kesegaran": 2.2, "cacat": 14.0},
    "Kasus 3 (kualitas tinggi)": {"panjang": 11.5, "kesegaran": 0.3, "cacat": 3.0},
}


def rp(x):
    return "Rp " + f"{x:,.0f}".replace(",", ".")


if __name__ == "__main__":
    fis = bangun_fis("min")
    fis_prod = bangun_fis("prod")

    # ---- Uraian tahap demi tahap untuk Kasus 2 --------------------------
    d2 = fis.compute(KASUS["Kasus 2 (kualitas sedang)"], return_detail=True)
    print("=" * 78)
    print("URAIAN PIPELINE FIS -- KASUS 2:", KASUS["Kasus 2 (kualitas sedang)"])
    print("=" * 78)
    print("Tahap 1 Fuzzifikasi :")
    for v, m in d2["derajat"].items():
        print(f"   {v:<10}", {l: round(x, 3) for l, x in m.items() if x > 0})
    print("Tahap 2 Evaluasi aturan (alpha > 0):")
    for rid, kons, a in d2["firing"]:
        if a > 0:
            print(f"   {rid:<4} -> Harga {kons:<7} alpha = {a:.3f}")
    print("Tahap 3 Agregasi per label:", {k: round(v, 3) for k, v in d2["alpha_label"].items()})
    print(f"Tahap 4 Defuzzifikasi centroid: {rp(d2['crisp'])}")

    # ---- Tiga kasus: output + perbandingan implikasi ---------------------
    print("\n" + "=" * 78)
    print("HASIL 3 KASUS UJI")
    print("=" * 78)
    print(f"{'Kasus':<28}{'Panjang':>8}{'Hari':>6}{'Cacat%':>8}{'Murah':>7}{'Sedang':>8}{'Mahal':>7}"
          f"{'Harga (Min)':>16}{'Harga (Prod)':>16}")
    detail = {}
    for nama, x in KASUS.items():
        dm = fis.compute(x, return_detail=True)
        dp = fis_prod.compute(x, return_detail=True)
        detail[nama] = dm
        al = dm["alpha_label"]
        print(f"{nama:<28}{x['panjang']:>8.1f}{x['kesegaran']:>6.1f}{x['cacat']:>8.1f}"
              f"{al['Murah']:>7.2f}{al['Sedang']:>8.2f}{al['Mahal']:>7.2f}"
              f"{rp(dm['crisp']):>16}{rp(dp['crisp']):>16}")

    # ---- Visualisasi pipeline (3 kasus) ---------------------------------
    z = kb.UNIVERSE["harga"]
    warna = {"Murah": "tab:green", "Sedang": "tab:blue", "Mahal": "tab:red"}
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), sharey=True)
    for ax, (nama, dm) in zip(axes, detail.items()):
        for lbl, w in warna.items():
            ax.plot(z, kb.mf_array("harga", lbl), "--", color=w, alpha=0.45, label=f"MF {lbl}")
            ax.fill_between(z, 0, dm["implikasi"][lbl], color=w, alpha=0.18)
        ax.fill_between(z, 0, dm["agregasi"], color="purple", alpha=0.30, label="Daerah agregasi")
        ax.plot(z, dm["agregasi"], color="purple", lw=2.2)
        ax.axvline(dm["crisp"], color="black", lw=2.2, label=f"Centroid = {rp(dm['crisp'])}")
        ax.set_title(nama, fontsize=11, fontweight="bold")
        ax.set_xlabel("Harga jual cabai (Rp/kg)")
        ax.set_ylim(-0.05, 1.08)
        ax.grid(True, ls=":", alpha=0.6)
        ax.legend(loc="upper center", fontsize=8, ncol=1)
    axes[0].set_ylabel("Derajat keanggotaan μ(z)")
    fig.suptitle("Pipeline FIS Mamdani: Implikasi, Agregasi, dan Defuzzifikasi Centroid", fontweight="bold")
    fig.tight_layout()
    fig.savefig("pipeline_fis_agregasi_defuzzifikasi.png", dpi=200)
    plt.close(fig)

    # ---- Min (clipping) vs Product (scaling) pada Kasus 2 ----------------
    dp2 = fis_prod.compute(KASUS["Kasus 2 (kualitas sedang)"], return_detail=True)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(z, d2["agregasi"], color="tab:blue", lw=2.2, label=f"Clipping/Min  (centroid {rp(d2['crisp'])})")
    ax.plot(z, dp2["agregasi"], color="tab:orange", lw=2.2, label=f"Scaling/Product (centroid {rp(dp2['crisp'])})")
    ax.set_xlabel("Harga jual cabai (Rp/kg)")
    ax.set_ylabel("μ agregasi")
    ax.set_title("Perbandingan Implikasi Min vs Product (Kasus 2)", fontweight="bold")
    ax.grid(True, ls=":", alpha=0.6)
    ax.legend()
    fig.tight_layout()
    fig.savefig("perbandingan_implikasi_min_vs_prod.png", dpi=200)
    plt.close(fig)

    # ---- Fallback ketika semua alpha = 0 (soal analisis 3) --------------
    print("\n" + "=" * 78)
    print("UJI FALLBACK: SEMUA ALPHA = 0")
    print("=" * 78)
    # basis aturan sengaja DIPOTONG: hanya aturan 'Mahal' (R25, R26, R22, R16)
    aturan_terbatas = [r for r in kb.daftar_aturan() if r[2] == "Mahal"]
    fis_parsial = bangun_fis("min", aturan_terbatas)
    uji = fis_parsial.compute(KASUS["Kasus 1 (kualitas rendah)"], return_detail=True)
    print(f"Basis aturan parsial: {len(aturan_terbatas)} aturan (hanya konsekuen Mahal)")
    print(f"Input Kasus 1 -> sum(mu_agregasi) = {uji['agregasi'].sum():.1f}, fallback dipakai = {uji['fallback']}")
    print(f"Output fallback (titik tengah semesta) = {rp(uji['crisp'])}")
    print("Tanpa fallback rumus sum(z*mu)/sum(mu) menjadi 0/0 (division by zero / NaN).")
    print("\nFile grafik: pipeline_fis_agregasi_defuzzifikasi.png, perbandingan_implikasi_min_vs_prod.png")
