import statistics as st
import time
import warnings
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    BaggingRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
    StackingRegressor,
    VotingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import (
    ARDRegression,
    BayesianRidge,
    ElasticNet,
    GammaRegressor,
    HuberRegressor,
    Lars,
    Lasso,
    LassoLars,
    LinearRegression,
    PassiveAggressiveRegressor,
    PoissonRegressor,
    RANSACRegressor,
    Ridge,
    SGDRegressor,
    TheilSenRegressor,
    TweedieRegressor,
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, RobustScaler, StandardScaler
from sklearn.svm import LinearSVR, NuSVR, SVR
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = BASE_DIR / "xor-dataset" / "artifacts_256" / "graph_statistics.csv"
DEFAULT_OUTPUT_DIR = BASE_DIR / "regression_results_v2_xor"
N_EXECUTIONS = 30
TEST_SIZE = 0.33
INCLUDE_SLOW_MODELS = False

# Portfólio de 20 regressores fortes e diversificados para dados tabulares.
# A seleção evita modelos redundantes ou usados apenas como baseline.
SELECTED_MODELS = [
    "RandomForest",
    "ExtraTrees",
    "GradientBoosting",
    "HistGradientBoosting",
    "AdaBoost",
    "BaggingDecisionTree",
    "VotingEnsemble",
    "StackingEnsemble",
    "LogTargetGradientBoosting",
    "SVRRBF",
    "NuSVR",
    "LinearSVR",
    "MLPRegressor",
    "KNNDistance",
    "Ridge",
    "ElasticNet",
    "BayesianRidge",
    "HuberRegressor",
    "PLSRegression",
    "PolynomialRidge",
]


def scaled(regressor):
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), regressor)


def robust_scaled(regressor):
    return make_pipeline(SimpleImputer(strategy="median"), RobustScaler(), regressor)


def tree_pipeline(regressor):
    return make_pipeline(SimpleImputer(strategy="median"), regressor)


def build_model_factories(include_slow=False):
    models = OrderedDict(
        [
            ("DummyMean", lambda seed: DummyRegressor(strategy="mean")),
            ("DummyMedian", lambda seed: DummyRegressor(strategy="median")),
            ("LinearRegression", lambda seed: scaled(LinearRegression())),
            ("Ridge", lambda seed: scaled(Ridge(alpha=1.0))),
            ("Lasso", lambda seed: scaled(Lasso(alpha=0.01, max_iter=10000))),
            (
                "ElasticNet",
                lambda seed: scaled(ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=10000)),
            ),
            ("Lars", lambda seed: scaled(Lars())),
            ("LassoLars", lambda seed: scaled(LassoLars(alpha=0.01))),
            ("BayesianRidge", lambda seed: scaled(BayesianRidge())),
            ("ARDRegression", lambda seed: scaled(ARDRegression())),
            ("HuberRegressor", lambda seed: robust_scaled(HuberRegressor(max_iter=1000))),
            (
                "RANSACRegressor",
                lambda seed: scaled(RANSACRegressor(random_state=seed)),
            ),
            (
                "SGDRegressor",
                lambda seed: scaled(
                    SGDRegressor(
                        loss="squared_error",
                        penalty="l2",
                        learning_rate="adaptive",
                        eta0=0.001,
                        max_iter=5000,
                        tol=1e-4,
                        early_stopping=True,
                        validation_fraction=0.1,
                        n_iter_no_change=20,
                        random_state=seed,
                    )
                ),
            ),
            (
                "PassiveAggressive",
                lambda seed: scaled(
                    PassiveAggressiveRegressor(max_iter=2000, random_state=seed, tol=1e-4)
                ),
            ),
            ("PoissonRegressor", lambda seed: scaled(PoissonRegressor(max_iter=1000))),
            ("GammaRegressor", lambda seed: scaled(GammaRegressor(max_iter=1000))),
            (
                "TweedieRegressor",
                lambda seed: scaled(TweedieRegressor(power=1.5, max_iter=1000)),
            ),
            (
                "PolynomialRidge",
                lambda seed: make_pipeline(
                    SimpleImputer(strategy="median"),
                    PolynomialFeatures(degree=2, include_bias=False),
                    StandardScaler(),
                    Ridge(alpha=1.0),
                ),
            ),
            (
                "DecisionTree",
                lambda seed: tree_pipeline(DecisionTreeRegressor(random_state=seed)),
            ),
            (
                "ExtraTree",
                lambda seed: tree_pipeline(ExtraTreeRegressor(random_state=seed)),
            ),
            (
                "RandomForest",
                lambda seed: tree_pipeline(
                    RandomForestRegressor(n_estimators=200, random_state=seed, n_jobs=-1)
                ),
            ),
            (
                "ExtraTrees",
                lambda seed: tree_pipeline(
                    ExtraTreesRegressor(n_estimators=200, random_state=seed, n_jobs=-1)
                ),
            ),
            (
                "GradientBoosting",
                lambda seed: tree_pipeline(GradientBoostingRegressor(random_state=seed)),
            ),
            (
                "HistGradientBoosting",
                lambda seed: tree_pipeline(HistGradientBoostingRegressor(random_state=seed)),
            ),
            (
                "AdaBoost",
                lambda seed: tree_pipeline(
                    AdaBoostRegressor(n_estimators=100, random_state=seed)
                ),
            ),
            (
                "BaggingDecisionTree",
                lambda seed: tree_pipeline(
                    BaggingRegressor(
                        estimator=DecisionTreeRegressor(),
                        n_estimators=100,
                        random_state=seed,
                        n_jobs=1,
                    )
                ),
            ),
            (
                "KNNUniform",
                lambda seed: scaled(KNeighborsRegressor(n_neighbors=5, weights="uniform", n_jobs=-1)),
            ),
            (
                "KNNDistance",
                lambda seed: scaled(KNeighborsRegressor(n_neighbors=5, weights="distance", n_jobs=-1)),
            ),
            ("LinearSVR", lambda seed: scaled(LinearSVR(random_state=seed, max_iter=10000))),
            ("SVRLinear", lambda seed: scaled(SVR(kernel="linear", cache_size=1000))),
            ("SVRRBF", lambda seed: scaled(SVR(kernel="rbf", cache_size=1000))),
            ("SVRPolynomial", lambda seed: scaled(SVR(kernel="poly", cache_size=1000))),
            ("NuSVR", lambda seed: scaled(NuSVR(cache_size=1000))),
            (
                "MLPRegressor",
                lambda seed: scaled(
                    MLPRegressor(
                        hidden_layer_sizes=(100, 50),
                        early_stopping=True,
                        max_iter=1000,
                        random_state=seed,
                    )
                ),
            ),
            ("PLSRegression", lambda seed: scaled(PLSRegression(n_components=2))),
            (
                "VotingEnsemble",
                lambda seed: tree_pipeline(
                    VotingRegressor(
                        estimators=[
                            ("rf", RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=-1)),
                            ("extra", ExtraTreesRegressor(n_estimators=100, random_state=seed, n_jobs=-1)),
                            ("gbr", GradientBoostingRegressor(random_state=seed)),
                        ],
                        n_jobs=1,
                    )
                ),
            ),
            (
                "StackingEnsemble",
                lambda seed: tree_pipeline(
                    StackingRegressor(
                        estimators=[
                            ("rf", RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=-1)),
                            ("extra", ExtraTreesRegressor(n_estimators=100, random_state=seed, n_jobs=-1)),
                            ("gbr", GradientBoostingRegressor(random_state=seed)),
                        ],
                        final_estimator=Ridge(),
                        cv=5,
                        n_jobs=1,
                    )
                ),
            ),
            (
                "LogTargetGradientBoosting",
                lambda seed: tree_pipeline(
                    TransformedTargetRegressor(
                        regressor=GradientBoostingRegressor(random_state=seed),
                        func=np.log1p,
                        inverse_func=np.expm1,
                    )
                ),
            ),
        ]
    )

    if include_slow:
        models["TheilSenRegressor"] = lambda seed: robust_scaled(
            TheilSenRegressor(random_state=seed, n_jobs=-1)
        )

    add_optional_models(models)
    return models


def add_optional_models(models):
    try:
        from xgboost import XGBRegressor

        models["XGBoost"] = lambda seed: tree_pipeline(
            XGBRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=6,
                objective="reg:squarederror",
                random_state=seed,
                n_jobs=-1,
            )
        )
    except ImportError:
        pass

    try:
        from lightgbm import LGBMRegressor

        models["LightGBM"] = lambda seed: tree_pipeline(
            LGBMRegressor(n_estimators=300, learning_rate=0.05, random_state=seed, n_jobs=-1)
        )
    except ImportError:
        pass

    try:
        from catboost import CatBoostRegressor

        models["CatBoost"] = lambda seed: tree_pipeline(
            CatBoostRegressor(iterations=300, learning_rate=0.05, verbose=False, random_seed=seed)
        )
    except ImportError:
        pass


def load_dataset(dataset_path):
    dataset = pd.read_csv(dataset_path, encoding="utf-8").replace([np.inf, -np.inf], np.nan)
    required_columns = {"Y", "snr", "gate"}
    missing_columns = required_columns.difference(dataset.columns)
    if missing_columns:
        raise ValueError(f"Colunas obrigatórias ausentes: {sorted(missing_columns)}")

    labels = pd.to_numeric(dataset["Y"], errors="coerce")
    valid_rows = labels.notna()
    dataset = dataset.loc[valid_rows].copy()
    labels = labels.loc[valid_rows]

    ignored_columns = ["snr", "Y", "gate", "circuit_id", "algorithm_mnemonic"]
    features = dataset.drop(columns=ignored_columns, errors="ignore")
    non_numeric_columns = list(features.select_dtypes(exclude=[np.number, "bool"]).columns)
    if non_numeric_columns:
        raise ValueError(f"Features não numéricas não tratadas: {non_numeric_columns}")

    features = features.astype(float)
    return features, labels.astype(float)


def build_scenarios(features, labels):
    return OrderedDict(
        [
            ("dataset_completo", (features, labels)),
            ("snr_menor_5", (features.loc[labels < 5], labels.loc[labels < 5])),
            ("snr_maior_igual_5", (features.loc[labels >= 5], labels.loc[labels >= 5])),
        ]
    )


def save_results(output_dir, detailed_results, summary_results, errors):
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(detailed_results).to_csv(output_dir / "resultados_execucoes.csv", index=False)
    pd.DataFrame(summary_results).to_csv(output_dir / "resultados_medios.csv", index=False)
    pd.DataFrame(errors, columns=["cenario", "algoritmo", "execucao", "erro"]).to_csv(
        output_dir / "erros.csv", index=False
    )


def run_experiments(features, labels, model_factories, executions, output_dir):
    detailed_results = []
    summary_results = []
    errors = []

    for scenario_name, (scenario_features, scenario_labels) in build_scenarios(features, labels).items():
        print(f"\n{'=' * 80}\nCenário: {scenario_name} ({len(scenario_labels)} amostras)\n{'=' * 80}")

        for model_name, model_factory in model_factories.items():
            mae_values = []
            mse_values = []
            rmse_values = []
            r2_values = []
            duration_values = []

            print(f"\nAlgoritmo: {model_name}")
            for execution in range(executions):
                started_at = time.perf_counter()
                try:
                    X_train, X_test, y_train, y_test = train_test_split(
                        scenario_features,
                        scenario_labels,
                        test_size=TEST_SIZE,
                        random_state=execution,
                    )
                    model = model_factory(execution)
                    model.fit(X_train, y_train)
                    predictions = np.asarray(model.predict(X_test)).reshape(-1)

                    mae = mean_absolute_error(y_test, predictions)
                    mse = mean_squared_error(y_test, predictions)
                    rmse = root_mean_squared_error(y_test, predictions)
                    r2 = r2_score(y_test, predictions)
                    duration = time.perf_counter() - started_at

                    mae_values.append(mae)
                    mse_values.append(mse)
                    rmse_values.append(rmse)
                    r2_values.append(r2)
                    duration_values.append(duration)
                    detailed_results.append(
                        {
                            "cenario": scenario_name,
                            "algoritmo": model_name,
                            "execucao": execution + 1,
                            "random_state": execution,
                            "n_amostras": len(scenario_labels),
                            "mae": mae,
                            "mse": mse,
                            "rmse": rmse,
                            "r2": r2,
                            "duracao_segundos": duration,
                        }
                    )
                    print(
                        f"run: {execution}, MAE: {mae:.6f}, MSE: {mse:.6f}, "
                        f"RMSE: {rmse:.6f}, R2: {r2:.6f}, tempo: {duration:.2f}s"
                    )
                except Exception as error:
                    duration = time.perf_counter() - started_at
                    errors.append((scenario_name, model_name, execution + 1, repr(error)))
                    print(f"run: {execution}, ERRO após {duration:.2f}s: {error}")

            if len(mae_values) != executions:
                print(
                    f"Resultado inválido: {model_name} concluiu {len(mae_values)}/{executions} execuções."
                )
                save_results(output_dir, detailed_results, summary_results, errors)
                continue

            summary = {
                "cenario": scenario_name,
                "algoritmo": model_name,
                "tipo": "ensemble" if "Ensemble" in model_name or model_name in {
                    "RandomForest",
                    "ExtraTrees",
                    "GradientBoosting",
                    "HistGradientBoosting",
                    "AdaBoost",
                    "BaggingDecisionTree",
                    "LogTargetGradientBoosting",
                    "XGBoost",
                    "LightGBM",
                    "CatBoost",
                } else "individual",
                "n_amostras": len(scenario_labels),
                "execucoes": executions,
                "mae_medio": st.mean(mae_values),
                "mse_medio": st.mean(mse_values),
                "rmse_medio": st.mean(rmse_values),
                "r2_medio": st.mean(r2_values),
                "mae_desvio_padrao": st.pstdev(mae_values),
                "mse_desvio_padrao": st.pstdev(mse_values),
                "rmse_desvio_padrao": st.pstdev(rmse_values),
                "r2_desvio_padrao": st.pstdev(r2_values),
                "tempo_medio_segundos": st.mean(duration_values),
            }
            summary_results.append(summary)
            print(
                f"Final (AVG {executions}) - MAE: {summary['mae_medio']:.6f}, "
                f"MSE: {summary['mse_medio']:.6f}, RMSE: {summary['rmse_medio']:.6f}, "
                f"R2: {summary['r2_medio']:.6f}"
            )
            save_results(output_dir, detailed_results, summary_results, errors)

    return detailed_results, summary_results, errors


model_factories = build_model_factories(include_slow=INCLUDE_SLOW_MODELS)

if SELECTED_MODELS is not None:
    unknown_models = [name for name in SELECTED_MODELS if name not in model_factories]
    if unknown_models:
        raise ValueError(
            f"Modelos desconhecidos: {unknown_models}. "
            f"Modelos disponíveis: {list(model_factories)}"
        )
    model_factories = OrderedDict(
        (name, model_factories[name]) for name in SELECTED_MODELS
    )

if N_EXECUTIONS < 1:
    raise ValueError("N_EXECUTIONS deve ser maior que zero.")

features, labels = load_dataset(DEFAULT_DATASET)
print(f"Dataset: {DEFAULT_DATASET}")
print(f"Amostras: {len(labels)} | Features: {features.shape[1]}")
print(f"Modelos: {len(model_factories)} | Execuções por modelo/cenário: {N_EXECUTIONS}")

detailed_results, summary_results, errors = run_experiments(
    features,
    labels,
    model_factories,
    N_EXECUTIONS,
    DEFAULT_OUTPUT_DIR,
)
save_results(DEFAULT_OUTPUT_DIR, detailed_results, summary_results, errors)
print(f"\nResultados salvos em: {DEFAULT_OUTPUT_DIR}")
