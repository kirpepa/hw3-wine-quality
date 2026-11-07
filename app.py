"""Демонстрация готовой модели качества вина; обучение выполняется в ноутбуке."""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn import metrics


APP_DIR = Path(__file__).resolve().parent
FEATURE_LABELS = {
    "fixed_acidity": "Фиксированная кислотность",
    "volatile_acidity": "Летучая кислотность",
    "citric_acid": "Лимонная кислота",
    "residual_sugar": "Остаточный сахар",
    "chlorides": "Хлориды",
    "free_sulfur_dioxide": "Свободный диоксид серы",
    "total_sulfur_dioxide": "Общий диоксид серы",
    "density": "Плотность",
    "pH": "pH",
    "sulphates": "Сульфаты",
    "alcohol": "Алкоголь, %",
    "is_red": "Красное вино",
}


@st.cache_resource(show_spinner=False)
def load_model(path, file_version):
    # Один объект модели на сервер: при rerun повторная десериализация не нужна.
    return joblib.load(path)


@st.cache_data(show_spinner=False)
def load_test_data(path, file_version):
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_config(path, file_version):
    return json.loads(Path(path).read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def test_proba(_model, X, model_version):
    # Порог не входит в ключ кэша: при его изменении вероятности не пересчитываются.
    # Версия файла инвалидирует кэш, если сохранённая модель была заменена.
    return _model.predict_proba(X)[:, 1]


def validate_features(table, feature_names):
    """Проверить CSV и вернуть числовые признаки в порядке обучения модели."""
    missing = [name for name in feature_names if name not in table.columns]
    if missing:
        raise ValueError("В CSV не хватает столбцов: " + ", ".join(missing))
    if table.empty:
        raise ValueError("CSV не содержит строк с винами.")
    try:
        features = table.loc[:, feature_names].apply(pd.to_numeric, errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError("Все признаки должны быть числами. Проверьте значения в CSV.") from exc
    if features.isna().any().any():
        raise ValueError("В признаках есть пустые значения. Заполните их перед загрузкой.")
    if features.isin([float("inf"), float("-inf")]).any().any():
        raise ValueError("Признаки не должны содержать бесконечные значения.")
    if not features["is_red"].isin([0, 1]).all():
        raise ValueError("Столбец is_red должен содержать только 0 (белое) и 1 (красное).")
    return features


st.set_page_config(page_title="Винная подборка", page_icon="🍷", layout="wide")
st.title("🍷 Винная подборка")
st.caption("Лабораторные характеристики → оценка качества → решение для подборки")

try:
    model_path = APP_DIR / "model.joblib"
    data_path = APP_DIR / "test.csv"
    config_path = APP_DIR / "config.json"
    model_version = model_path.stat().st_mtime_ns
    model = load_model(str(model_path), model_version)
    test = load_test_data(str(data_path), data_path.stat().st_mtime_ns)
    config = load_config(str(config_path), config_path.stat().st_mtime_ns)
    feature_names = list(model.feature_names_in_)
    X_test = validate_features(test, feature_names)
    y_test = test["target"]
    default_threshold = float(config["threshold"])
    if not 0 <= default_threshold <= 1:
        raise ValueError("Порог в config.json должен быть от 0 до 1.")
    if not y_test.isin([0, 1]).all() or y_test.nunique() != 2:
        raise ValueError("В test.csv должны присутствовать оба класса target: 0 и 1.")
    test_scores = test_proba(model, X_test, model_version)
except (OSError, ValueError, KeyError, AttributeError, EOFError) as exc:
    st.error(f"Не удалось загрузить данные приложения: {exc}")
    st.info("Сначала выполните ячейку сохранения модели в HW3_MMDA.ipynb.")
    st.stop()

with st.sidebar:
    st.header("Правило отбора")
    threshold = st.slider(
        "Минимальная вероятность хорошего вина",
        min_value=0.0,
        max_value=1.0,
        value=default_threshold,
        step=0.001,
        format="%.3f",
        key="threshold",
    )
    st.caption(f"Порог из ноутбука: {default_threshold:.4f}")
    st.write("Вино попадает в подборку, если вероятность **не ниже порога**.")
    st.info(
        "Повышение порога сокращает подборку и обычно повышает её точность. "
        "Снижение порога добавляет больше хороших вин, но также больше ошибочных рекомендаций."
    )

single_tab, metrics_tab, batch_tab, coefficients_tab = st.tabs(
    ["Одно вино", "Модельные метрики", "Пакетное предсказание", "Что выучила модель"]
)

with single_tab:
    st.subheader("Характеристики вина")
    st.caption("Диапазоны и исходные значения взяты из test.csv; исходное значение — медиана.")
    columns = st.columns(3)
    values = {}
    for index, feature in enumerate(feature_names):
        with columns[index % 3]:
            if feature == "is_red":
                values[feature] = int(st.toggle("Красное вино", value=bool(X_test[feature].median())))
                st.caption("Выключено — белое вино")
            else:
                values[feature] = st.number_input(
                    FEATURE_LABELS.get(feature, feature),
                    min_value=float(X_test[feature].min()),
                    max_value=float(X_test[feature].max()),
                    value=float(X_test[feature].median()),
                    format="%.4f",
                    key=f"feature_{feature}",
                    help=f"Столбец CSV: {feature}",
                )
    sample = pd.DataFrame([values], columns=feature_names)
    probability = float(test_proba(model, sample, model_version)[0])
    st.metric("Вероятность хорошего вина", f"{probability:.1%}")
    if probability >= threshold:
        st.success(f"Включить в подборку: {probability:.3f} ≥ {threshold:.3f}.")
    else:
        st.warning(f"Не включать в подборку: {probability:.3f} < {threshold:.3f}.")
    st.caption("Хорошее вино в этой задаче — экспертная оценка не ниже 7 из 10.")

with metrics_tab:
    st.subheader("Качество отбора при текущем пороге")
    predictions = (test_scores >= threshold).astype(int)
    precision = metrics.precision_score(y_test, predictions, zero_division=0)
    recall = metrics.recall_score(y_test, predictions, zero_division=0)
    f1 = metrics.f1_score(y_test, predictions, zero_division=0)
    tn, fp, fn, tp = metrics.confusion_matrix(y_test, predictions, labels=[0, 1]).ravel()
    cards = st.columns(4)
    cards[0].metric("Precision · точность подборки", f"{precision:.1%}")
    cards[1].metric("Recall · найдено хороших вин", f"{recall:.1%}")
    cards[2].metric("F1", f"{f1:.3f}")
    cards[3].metric("В подборке", f"{int(predictions.sum())} / {len(predictions)}")
    if not predictions.any():
        st.info("При этом пороге подборка пуста. Для расчёта метрик precision принят равным 0.")

    matrix_col, explanation_col = st.columns([1, 1])
    with matrix_col:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        metrics.ConfusionMatrixDisplay.from_predictions(
            y_test,
            predictions,
            labels=[0, 1],
            display_labels=["Остальные", "Хорошие"],
            ax=ax,
            colorbar=False,
            cmap="Purples",
        )
        ax.set(xlabel="Предсказание модели", ylabel="Экспертная оценка", title="Матрица ошибок")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    with explanation_col:
        st.write(f"**{tp} хороших вин** отобраны, **{fn} хороших вин** пропущены.")
        st.write(f"**{fp} остальных вин** ошибочно рекомендованы, **{tn}** отклонены правильно.")
        st.write(
            "Precision отвечает на вопрос: «Какая доля рекомендованных вин действительно хорошая?». "
            "Recall показывает, какую долю всех хороших вин удалось найти."
        )
        st.caption(
            "Ползунок помогает изучить компромисс. Для итоговой оценки порог был выбран "
            "на валидации в ноутбуке; подбирать его по этой тестовой выборке не следует."
        )

    fpr, tpr, _ = metrics.roc_curve(y_test, test_scores)
    curve_precision, curve_recall, _ = metrics.precision_recall_curve(y_test, test_scores)
    roc_auc = metrics.roc_auc_score(y_test, test_scores)
    average_precision = metrics.average_precision_score(y_test, test_scores)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(fpr, tpr, color="#713656", label=f"ROC-AUC = {roc_auc:.3f}")
    axes[0].plot([0, 1], [0, 1], "--", color="gray", label="Случайное ранжирование")
    axes[0].scatter(fp / (fp + tn), recall, color="#df9744", zorder=3, label=f"Порог {threshold:.3f}")
    axes[0].set(xlabel="Доля ложных рекомендаций (FPR)", ylabel="Recall (TPR)", title="ROC-кривая")
    axes[1].plot(curve_recall, curve_precision, color="#713656", label=f"PR-AUC (AP) = {average_precision:.3f}")
    axes[1].axhline(y_test.mean(), linestyle="--", color="gray", label=f"Доля хороших вин: {y_test.mean():.3f}")
    axes[1].scatter(recall, precision, color="#df9744", zorder=3, label=f"Порог {threshold:.3f}")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="PR-кривая")
    for ax in axes:
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.caption("Кривые и площади под ними не зависят от порога. Меняется только отмеченная точка.")

with batch_tab:
    st.subheader("Загрузите характеристики нескольких вин")
    st.write("CSV с заголовком, разделитель — запятая, десятичный разделитель — точка. Столбец target не нужен.")
    st.code(",".join(feature_names), language=None)
    st.download_button(
        "Скачать пример CSV",
        data=X_test.head(5).to_csv(index=False).encode("utf-8"),
        file_name="wine_example.csv",
        mime="text/csv",
    )
    uploaded = st.file_uploader("CSV с признаками вин", type="csv")
    if uploaded is not None:
        try:
            table = pd.read_csv(uploaded)
            features = validate_features(table, feature_names)
            extra = [name for name in table.columns if name not in feature_names]
            if extra:
                st.info("При прогнозировании игнорируются дополнительные столбцы: " + ", ".join(extra))
            batch_scores = test_proba(model, features, model_version)
            result = features.assign(
                probability=batch_scores,
                prediction=(batch_scores >= threshold).astype(int),
            )
            st.caption(f"Общий порог: {threshold:.3f}; prediction=1 — включить в подборку.")
            st.dataframe(result, hide_index=True)
            st.download_button(
                "Скачать прогнозы CSV",
                data=result.to_csv(index=False).encode("utf-8"),
                file_name="wine_predictions.csv",
                mime="text/csv",
            )
        except (ValueError, TypeError, UnicodeError, pd.errors.ParserError) as exc:
            st.error(f"Не удалось обработать CSV: {exc}")

with coefficients_tab:
    st.subheader("Какие признаки влияют на оценку")
    coefficients = pd.Series(model.steps[-1][1].coef_[0], index=feature_names).sort_values()
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#713656" if coefficient > 0 else "#4d7d85" for coefficient in coefficients]
    ax.barh([FEATURE_LABELS.get(name, name) for name in coefficients.index], coefficients, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set(xlabel="Коэффициент после StandardScaler", title="Влияние увеличения признака на одно стандартное отклонение")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    positive = coefficients[coefficients > 0].nlargest(3)
    negative = coefficients[coefficients < 0].nsmallest(3)
    st.write("**Сильнее голосуют за хорошее вино:** " + ", ".join(FEATURE_LABELS.get(name, name) for name in positive.index) + ".")
    st.write("**Сильнее голосуют против:** " + ", ".join(FEATURE_LABELS.get(name, name) for name in negative.index) + ".")
    st.caption(
        "Коэффициенты меняют логарифм отношения шансов при прочих равных. "
        "Это связи, найденные моделью, а не доказательства причинного влияния."
    )
