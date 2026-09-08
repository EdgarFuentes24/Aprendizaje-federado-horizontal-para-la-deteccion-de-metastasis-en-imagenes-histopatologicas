"""
main.py  –  Experimentos FedHEONN sobre Camelyon17-WILDS

Ejecuta:
    - Centralizado (normalización global)
    - Centralizado (normalización local)
    - Federado sin cifrado
    - Federado con cifrado homomórfico

Los resultados se guardan en results/all_results.xlsx.
"""

import time

from src.config import (
    ARCHITECTURES,
    FEATURES_DIR,
    HOSPITAL_IDS,
    LAMBDAS,
    VERSION,
)

from src.utils import load_features, append_results

from src.centralized import (
    train_centralized,
    evaluate_centralized,
)

from src.federated import (
    train_federated,
    evaluate_federated,
)


# ---------------------------------------------------------------------------
# Carga de datos
# ---------------------------------------------------------------------------

def load_split(architecture, split, version=VERSION):

    hospitals = []

    for hospital_id in HOSPITAL_IDS[split]:
        path = (
            f"{FEATURES_DIR}/"
            f"{architecture}_hospital{hospital_id}_features{version}.pt"
        )
        hospitals.append(load_features(path))

    return hospitals


# ---------------------------------------------------------------------------
# Presentación
# ---------------------------------------------------------------------------

def print_metrics(split, metrics):

    auc = (
        f"{metrics['auc']:.2f}%"
        if metrics["auc"] is not None
        else "N/A"
    )

    print(
        f"    {split:<5} │ "
        f"Acc={metrics['acc']:.2f}%  "
        f"AUC={auc}  "
        f"F1={metrics['f1']:.2f}%  "
        f"Sens={metrics['sensitivity']:.2f}%  "
        f"Spec={metrics['specificity']:.2f}%"
    )


def build_results(
    architecture,
    mode,
    normalization,
    lam,
    val_m,
    test_m,
    t_train,
):

    return {
        "Arquitectura": architecture,
        "Modo": mode,
        "Normalización": normalization,
        "lambda": lam,

        "val_acc": val_m["acc"],
        "val_auc": val_m["auc"],
        "val_f1": val_m["f1"],
        "val_sensitivity": val_m["sensitivity"],
        "val_specificity": val_m["specificity"],

        "test_acc": test_m["acc"],
        "test_auc": test_m["auc"],
        "test_f1": test_m["f1"],
        "test_sensitivity": test_m["sensitivity"],
        "test_specificity": test_m["specificity"],

        "time_sec": t_train,
    }


# ---------------------------------------------------------------------------
# Centralizado
# ---------------------------------------------------------------------------

def run_centralized(
    architecture,
    train_data,
    val_data,
    test_data,
    normalization,
    lam,
):

    print(
        f"\n  [Centralizado | "
        f"norm={normalization} | "
        f"λ={lam}]"
    )

    t0 = time.time()

    model, scaler_global = train_centralized(
        train_data,
        normalization,
        lam,
    )

    t_train = time.time() - t0

    val_metrics = evaluate_centralized(
        model,
        val_data[0],
        normalization,
        scaler_global,
    )

    test_metrics = evaluate_centralized(
        model,
        test_data[0],
        normalization,
        scaler_global,
    )

    print_metrics("Val", val_metrics)
    print_metrics("Test", test_metrics)
    print(f"    Tiempo : {t_train:.1f}s")

    return build_results(
        architecture,
        "centralizado",
        normalization,
        lam,
        val_metrics,
        test_metrics,
        t_train,
    )


# ---------------------------------------------------------------------------
# Federado
# ---------------------------------------------------------------------------

def run_federated(
    architecture,
    train_data,
    val_data,
    test_data,
    lam,
    encrypted=False,
):

    encryption_label = "con encriptacion" if encrypted else "sin encriptacion"

    print(
        f"\n  [Federado | "
        f"{encryption_label} | "
        f"λ={lam}]"
    )

    t0 = time.time()

    coordinator = train_federated(
        train_data,
        lam,
        encrypted=encrypted,
    )

    t_train = time.time() - t0

    val_metrics = evaluate_federated(
        coordinator,
        val_data[0],
        encrypted=encrypted,
    )

    test_metrics = evaluate_federated(
        coordinator,
        test_data[0],
        encrypted=encrypted,
    )

    print_metrics("Val", val_metrics)
    print_metrics("Test", test_metrics)
    print(f"    Tiempo : {t_train:.1f}s")

    return build_results(
        architecture,
        "federado",
        encryption_label,
        lam,
        val_metrics,
        test_metrics,
        t_train,
    )


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():

    for architecture in ARCHITECTURES:

        print(f"\n{'=' * 60}")
        print(f" Arquitectura: {architecture}")
        print(f"{'=' * 60}")

        train_data = load_split(architecture, "train")
        val_data = load_split(architecture, "val")
        test_data = load_split(architecture, "test")

        for lam in LAMBDAS:

            print(f"\n──── λ = {lam} ────")

            # -------------------------------------------------------------
            # Centralizado global
            # -------------------------------------------------------------

#            results = run_centralized(
#                architecture,
#                train_data,
#                val_data,
#                test_data,
#                normalization="global",
#                lam=lam,
#            )

#            append_results(results)

            # -------------------------------------------------------------
            # Centralizado local
            # -------------------------------------------------------------

#            results = run_centralized(
#                architecture,
#                train_data,
#                val_data,
#                test_data,
#                normalization="local",
#                lam=lam,
#            )

#            append_results(results)

            # -------------------------------------------------------------
            # Federado sin cifrado
            # -------------------------------------------------------------

            results = run_federated(
                architecture,
                train_data,
                val_data,
                test_data,
                lam=lam,
                encrypted=False,
            )

            append_results(results)

            # -------------------------------------------------------------
            # Federado con cifrado homomórfico
            # -------------------------------------------------------------

            results = run_federated(
                architecture,
                train_data,
                val_data,
                test_data,
                lam=lam,
                encrypted=True,
            )

            append_results(results)


if __name__ == "__main__":
    main()