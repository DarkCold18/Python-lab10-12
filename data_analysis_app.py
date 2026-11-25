"""
Програма комплексного аналізу даних продажів із використанням декількох бібліотек Python.
Стек бібліотек: NumPy, Pandas, SciPy, Matplotlib, Seaborn, OpenPyXL.
Архітектура: об'єктно-орієнтована (класи DataLoader, DataAnalyzer, DataVisualizer).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import matplotlib
# Використовуємо неінтерактивний бекенд Agg для надійного автозбереження PNG без зависання вікон
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

# КЛАС 1: ЗАВАНТАЖЕННЯ, ОЧИЩЕННЯ ТА МУЛЬТИФОРМАТНИЙ ЕКСПОРТ ДАНИХ
class DataLoader:
    """
    Клас відповідає за генерацію тестових наборів, імпорт даних із файлів
    форматів CSV, JSON та Excel, їх очищення та експорт результатів
    """
    REQUIRED_COLUMNS: tuple[str, ...] = (
        "date",
        "category",
        "region",
        "units_sold",
        "unit_price",
        "discount_pct",
        "ad_spend",
        "customer_rating",
    )

    NUMERIC_COLUMNS: list[str] = [
        "units_sold",
        "unit_price",
        "discount_pct",
        "ad_spend",
        "customer_rating",
    ]

    def __init__(self) -> None:
        # Інкапсульовані  атрибути об'єкта
        self._file_path: Optional[Path] = None
        self._raw_df: Optional[pd.DataFrame] = None
        self._clean_df: Optional[pd.DataFrame] = None
        self._cleaning_report: dict[str, int] = {}

    @property
    def raw_data(self) -> Optional[pd.DataFrame]:
        """Повертає копію сирих даних для збереження інкапсуляції."""
        return self._raw_df.copy() if self._raw_df is not None else None

    @property
    def clean_data(self) -> Optional[pd.DataFrame]:
        """Повертає копію очищених даних."""
        return self._clean_df.copy() if self._clean_df is not None else None

    @property
    def cleaning_report(self) -> dict[str, int]:
        """Повертає статистику етапу очищення."""
        return dict(self._cleaning_report)

    def generate_sample_file(
        self, filepath: str, fmt: str = "csv", n_rows: int = 80
    ) -> str:
        """
        Генерує реалістичний набір даних про продажі з навмисними пропусками,
        дублікатами та аномальним викидом для демонстрації очищення
        """
        rng = np.random.default_rng(seed=42)
        categories = ["Електроніка", "Побутова техніка", "Комп'ютери", "Аксесуари"]
        regions = ["Київ", "Львів", "Дніпро", "Одеса", "Харків"]
        # Базові ціни залежно від категорії товару
        base_prices = {
            "Електроніка": 11500.0,
            "Побутова техніка": 9200.0,
            "Комп'ютери": 24500.0,
            "Аксесуари": 1800.0,
        }

        dates = pd.date_range(start="2026-01-01", periods=n_rows, freq="D")
        chosen_categories = rng.choice(categories, size=n_rows, p=[0.30, 0.25, 0.20, 0.25])
        chosen_regions = rng.choice(regions, size=n_rows)

        # Моделюємо позитивну кореляцію між рекламними витратами та кількістю продажів
        ad_spend = np.round(rng.uniform(3000.0, 25000.0, size=n_rows), 2)
        units_sold = np.maximum(
            1,
            np.round(ad_spend / 1400.0 + rng.normal(4.0, 2.5, size=n_rows)),
        ).astype(float)

        unit_price = np.array(
            [
                round(base_prices[cat] * rng.uniform(0.85, 1.18), 2)
                for cat in chosen_categories
            ],
            dtype=float,
        )
        discount_pct = rng.choice([0.0, 5.0, 10.0, 15.0, 20.0], size=n_rows)
        customer_rating = np.round(np.clip(rng.normal(4.4, 0.45, size=n_rows), 2.5, 5.0), 1)

        df = pd.DataFrame(
            {
                "date": dates.strftime("%Y-%m-%d"),
                "category": chosen_categories,
                "region": chosen_regions,
                "units_sold": units_sold,
                "unit_price": unit_price,
                "discount_pct": discount_pct,
                "ad_spend": ad_spend,
                "customer_rating": customer_rating,
            }
        )

        # Додаємо навмисні дефекти якості даних для демонстрації очищення:
        # 1) Пропущені значення (NaN)
        df.loc[5, "units_sold"] = np.nan
        df.loc[18, "unit_price"] = np.nan
        df.loc[33, "customer_rating"] = np.nan
        df.loc[50, "ad_spend"] = np.nan

        # 2) Статистичний викид (помилково введене число 450 одиниць за один день)
        df.loc[12, "units_sold"] = 450.0

        # 3) Два повні дублікати рядків
        df = pd.concat([df, df.iloc[[0, 1]]], ignore_index=True)

        out_path = Path(filepath)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fmt_clean = fmt.lower().strip().lstrip(".")

        try:
            if fmt_clean == "csv":
                df.to_csv(out_path, index=False, encoding="utf-8-sig")
            elif fmt_clean == "json":
                df.to_json(out_path, orient="records", force_ascii=False, indent=2)
            elif fmt_clean == "xlsx":
                df.to_excel(out_path, index=False, engine="openpyxl")
            else:
                raise ValueError(
                    f"Непідтримуваний формат генерації: '{fmt}'. Доступні: csv, json, xlsx."
                )
        except ImportError as exc:
            raise ImportError(
                "Для роботи з Excel-файлами встановіть бібліотеку openpyxl: pip install openpyxl"
            ) from exc

        return str(out_path)

    def load_data(self, filepath: str) -> pd.DataFrame:
        """
        Імпортує табличні дані з файлу CSV, JSON або Excel із валідацією
        існування файлу, формату та наявності обов'язкових стовпців.
        """
        path = Path(filepath)
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"Файл не знайдено за шляхом: '{filepath}'.")

        ext = path.suffix.lower()

        try:
            if ext == ".csv":
                df = pd.read_csv(path, encoding="utf-8-sig")
            elif ext == ".json":
                with open(path, "r", encoding="utf-8") as f:
                    raw_json = json.load(f)
                df = pd.DataFrame(raw_json)
            elif ext in (".xlsx", ".xls"):
                df = pd.read_excel(path, engine="openpyxl")
            else:
                raise ValueError(
                    f"Непідтримуваний формат файлу '{ext}'. Підтримуються: .csv, .json, .xlsx."
                )
        except pd.errors.EmptyDataError as exc:
            raise ValueError("Файл порожній і не містить табличних даних.") from exc
        except json.JSONDecodeError as exc:
            raise ValueError(f"Помилка структури JSON у файлі: {exc.msg}.") from exc
        except ImportError as exc:
            raise ImportError(
                "Відсутній рушій openpyxl для читання Excel. Виконайте: pip install openpyxl"
            ) from exc

        if df.empty:
            raise ValueError("Завантажена таблиця не містить жодного запису.")

        # Перевірка наявності обов'язкових стовпців
        missing_cols = [col for col in self.REQUIRED_COLUMNS if col not in df.columns]
        if missing_cols:
            raise KeyError(
                f"У файлі відсутні обов'язкові стовпці: {', '.join(missing_cols)}."
            )

        self._file_path = path
        self._raw_df = df.copy()
        self._clean_df = None
        return self._raw_df.copy()

    def clean_data_pipeline(
        self,
        missing_strategy: str = "median",
        remove_outliers: bool = True,
        z_threshold: float = 3.0,
    ) -> pd.DataFrame:
        """
        Послідовно виконує очищення даних:
        1) видалення дублікатів;
        2) конвертація дат і числових типів;
        3) заповнення або видалення пропусків;
        4) фільтрація статистичних викидів за Z-оцінкою (NumPy);
        5) обчислення похідних фінансових стовпців (revenue, net_revenue).
        """
        if self._raw_df is None:
            raise RuntimeError("Дані не завантажено. Спочатку виконайте імпорт файлу.")

        if missing_strategy not in ("median", "mean", "drop"):
            raise ValueError(
                "Некоректна стратегія для пропусків. Оберіть: 'median', 'mean' або 'drop'."
            )

        if z_threshold <= 0:
            raise ValueError("Поріг Z-оцінки має бути додатним числом.")

        df = self._raw_df.copy()
        initial_rows = len(df)

        # Крок 1. Видалення повних дублікатів
        df = df.drop_duplicates().copy()
        duplicates_removed = initial_rows - len(df)

        # Крок 2. Приведення типів (дати та числові поля)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        invalid_dates = int(df["date"].isna().sum())
        df = df.dropna(subset=["date"]).copy()

        for col in self.NUMERIC_COLUMNS:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Крок 3. Оброблення пропущених значень
        missing_before = int(df[self.NUMERIC_COLUMNS].isna().sum().sum())
        if missing_strategy == "drop":
            df = df.dropna(subset=self.NUMERIC_COLUMNS).copy()
        elif missing_strategy == "median":
            for col in self.NUMERIC_COLUMNS:
                df[col] = df[col].fillna(df[col].median())
        elif missing_strategy == "mean":
            for col in self.NUMERIC_COLUMNS:
                df[col] = df[col].fillna(df[col].mean())

        # Крок 4. Виявлення та фільтрація викидів за методом Z-оцінки (NumPy)
        outliers_removed = 0
        if remove_outliers and len(df) >= 5:
            mask = np.ones(len(df), dtype=bool)
            for col in ("units_sold", "unit_price"):
                values = df[col].to_numpy(dtype=float)
                std_val = float(np.std(values))
                if std_val > 0:
                    z_scores = np.abs((values - float(np.mean(values))) / std_val)
                    mask &= z_scores < z_threshold
            outliers_removed = int(len(df) - np.sum(mask))
            df = df.loc[mask].copy()

        # Крок 5. Обчислення похідних аналітичних показників
        df["revenue"] = np.round(df["units_sold"] * df["unit_price"], 2)
        df["net_revenue"] = np.round(
            df["revenue"] * (1.0 - df["discount_pct"] / 100.0), 2
        )
        df = df.sort_values("date").reset_index(drop=True)

        self._clean_df = df
        self._cleaning_report = {
            "initial_rows": initial_rows,
            "duplicates_removed": duplicates_removed,
            "invalid_dates_removed": invalid_dates,
            "missing_values_handled": missing_before,
            "outliers_removed": outliers_removed,
            "final_rows": len(df),
        }
        return self._clean_df.copy()

    def export_data(
        self,
        output_dir: str = "output",
        base_name: str = "cleaned_sales",
        formats: Optional[list[str]] = None,
    ) -> list[str]:
        """
        Зберігає очищений набір даних у декількох форматах (CSV, JSON, Excel).
        """
        if self._clean_df is None:
            raise RuntimeError("Немає очищених даних для експорту.")

        if formats is None:
            formats = ["csv", "json", "xlsx"]

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        saved_paths: list[str] = []

        # Підготовка копії з рядковим форматом дати для коректної серіалізації
        export_df = self._clean_df.copy()
        export_df["date"] = export_df["date"].dt.strftime("%Y-%m-%d")

        for fmt in formats:
            fmt_lower = fmt.lower().strip()
            try:
                if fmt_lower == "csv":
                    file_path = out_dir / f"{base_name}.csv"
                    export_df.to_csv(file_path, index=False, encoding="utf-8-sig")
                elif fmt_lower == "json":
                    file_path = out_dir / f"{base_name}.json"
                    export_df.to_json(
                        file_path, orient="records", force_ascii=False, indent=2
                    )
                elif fmt_lower == "xlsx":
                    file_path = out_dir / f"{base_name}.xlsx"
                    export_df.to_excel(file_path, index=False, engine="openpyxl")
                else:
                    print(f"[УВАГА] Формат '{fmt}' пропущено (не підтримується).")
                    continue
                saved_paths.append(str(file_path))
            except PermissionError:
                print(
                    f"[ПОМИЛКА] Немає доступу до запису файлу '{base_name}.{fmt_lower}'. "
                    "Можливо, файл відкрито в іншій програмі."
                )
            except ImportError:
                print(
                    "[ПОМИЛКА] Для експорту в Excel потрібна бібліотека openpyxl "
                    "(pip install openpyxl)."
                )

        return saved_paths

# КЛАС 2: СТАТИСТИЧНИЙ ТА АНАЛІТИЧНИЙ МОДУЛЬ
class DataAnalyzer:
    """
    Клас виконує описову статистику, групування за категоріями,
    кореляційний аналіз, перевірку нормальності та лінійну регресію.
    """

    ANALYSIS_COLUMNS: list[str] = [
        "units_sold",
        "unit_price",
        "discount_pct",
        "ad_spend",
        "customer_rating",
        "net_revenue",
    ]

    def __init__(self, df: pd.DataFrame) -> None:
        if df is None or df.empty:
            raise ValueError("Неможливо ініціалізувати DataAnalyzer із порожньою таблицею.")
        self._df: pd.DataFrame = df.copy()
        self._summary_stats: Optional[pd.DataFrame] = None
        self._category_stats: Optional[pd.DataFrame] = None
        self._corr_matrix: Optional[pd.DataFrame] = None
        self._stat_tests: dict[str, Any] = {}

    @property
    def summary_stats(self) -> Optional[pd.DataFrame]:
        return self._summary_stats.copy() if self._summary_stats is not None else None

    @property
    def category_stats(self) -> Optional[pd.DataFrame]:
        return self._category_stats.copy() if self._category_stats is not None else None

    @property
    def corr_matrix(self) -> Optional[pd.DataFrame]:
        return self._corr_matrix.copy() if self._corr_matrix is not None else None

    @property
    def stat_tests(self) -> dict[str, Any]:
        return dict(self._stat_tests)

    def compute_descriptive_stats(self) -> pd.DataFrame:
        """
        Розраховує розширені описові статистики засобами Pandas та NumPy:
        середнє, медіану, стандартне відхилення, мінімум, максимум, Q1, Q3 та IQR.
        """
        records: list[dict[str, Any]] = []
        for col in self.ANALYSIS_COLUMNS:
            arr = self._df[col].to_numpy(dtype=float)
            q1 = float(np.percentile(arr, 25))
            q3 = float(np.percentile(arr, 75))
            records.append(
                {
                    "Показник": col,
                    "Середнє": round(float(np.mean(arr)), 2),
                    "Медіана": round(float(np.median(arr)), 2),
                    "Стд. відхил.": round(float(np.std(arr, ddof=1)), 2),
                    "Мін": round(float(np.min(arr)), 2),
                    "Макс": round(float(np.max(arr)), 2),
                    "IQR (Q3-Q1)": round(q3 - q1, 2),
                }
            )
        self._summary_stats = pd.DataFrame(records)
        return self._summary_stats.copy()

    def analyze_by_category(self) -> pd.DataFrame:
        """
        Виконує групування за категоріями товарів (Split-Apply-Combine).
        """
        grouped = (
            self._df.groupby("category", as_index=False)
            .agg(
                transactions=("date", "count"),
                total_units=("units_sold", "sum"),
                total_net_revenue=("net_revenue", "sum"),
                avg_net_revenue=("net_revenue", "mean"),
                avg_rating=("customer_rating", "mean"),
            )
            .sort_values("total_net_revenue", ascending=False)
            .reset_index(drop=True)
        )
        grouped["total_net_revenue"] = grouped["total_net_revenue"].round(2)
        grouped["avg_net_revenue"] = grouped["avg_net_revenue"].round(2)
        grouped["avg_rating"] = grouped["avg_rating"].round(2)

        self._category_stats = grouped
        return self._category_stats.copy()

    def compute_correlations(self) -> pd.DataFrame:
        """
        Обчислює матрицю кореляцій Пірсона для всіх числових змінних.
        """
        self._corr_matrix = self._df[self.ANALYSIS_COLUMNS].corr(method="pearson").round(3)
        return self._corr_matrix.copy()

    def run_statistical_tests(self) -> dict[str, Any]:
        """
        Виконує інференційний статистичний аналіз засобами SciPy:
        1) Тест Шапіро–Уїлка на нормальність розподілу чистої виручки;
        2) Кореляція Пірсона та лінійна регресія між витратами на рекламу
           (ad_spend) і чистою виручкою (net_revenue).
        """
        net_rev = self._df["net_revenue"].to_numpy(dtype=float)
        ad_spend = self._df["ad_spend"].to_numpy(dtype=float)

        results: dict[str, Any] = {}

        # Перевірка нормальності розподілу (потрібно щонайменше 3 спостереження)
        if len(net_rev) >= 3:
            shapiro_stat, shapiro_p = stats.shapiro(net_rev)
            results["shapiro_stat"] = round(float(shapiro_stat), 4)
            results["shapiro_p"] = round(float(shapiro_p), 5)
            results["is_normal_alpha_05"] = bool(shapiro_p > 0.05)
        else:
            results["shapiro_stat"] = None
            results["shapiro_p"] = None
            results["is_normal_alpha_05"] = False

        # Лінійна регресія та значущість кореляції
        if len(ad_spend) >= 2 and np.std(ad_spend) > 0 and np.std(net_rev) > 0:
            reg = stats.linregress(ad_spend, net_rev)
            r_squared = float(reg.rvalue ** 2)
            results.update(
                {
                    "slope": round(float(reg.slope), 4),
                    "intercept": round(float(reg.intercept), 2),
                    "pearson_r": round(float(reg.rvalue), 4),
                    "p_value": round(float(reg.pvalue), 6),
                    "r_squared": round(r_squared, 4),
                }
            )
        else:
            raise ValueError(
                "Недостатньо варіативності в даних для побудови лінійної регресії."
            )

        self._stat_tests = results
        return dict(self._stat_tests)

    def save_text_report(
        self, output_path: str = "output/analysis_report.txt", cleaning_report: Optional[dict[str, int]] = None
    ) -> str:
        """
        Формує та зберігає структурований текстовий звіт із результатами аналізу.
        """
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        lines: list[str] = [
            "=" * 78,
            "ПІДСУМКОВИЙ ЗВІТ КОМПЛЕКСНОГО АНАЛІЗУ ДАНИХ ПРОДАЖІВ",
            "=" * 78,
            "",
        ]

        if cleaning_report:
            lines.append("1. РЕЗУЛЬТАТИ ОЧИЩЕННЯ ДАНИХ:")
            lines.append(f"   - Початкова кількість записів:     {cleaning_report.get('initial_rows', 0)}")
            lines.append(f"   - Видалено дублікатів:             {cleaning_report.get('duplicates_removed', 0)}")
            lines.append(f"   - Видалено некоректних дат:        {cleaning_report.get('invalid_dates_removed', 0)}")
            lines.append(f"   - Оброблено пропущених значень:    {cleaning_report.get('missing_values_handled', 0)}")
            lines.append(f"   - Відфільтровано викидів (Z-score):{cleaning_report.get('outliers_removed', 0)}")
            lines.append(f"   - Залишилось записів для аналізу:  {cleaning_report.get('final_rows', 0)}")
            lines.append("")

        if self._summary_stats is not None:
            lines.append("2. ОПИСОВА СТАТИСТИКА (NumPy + Pandas):")
            lines.append(self._summary_stats.to_string(index=False))
            lines.append("")

        if self._category_stats is not None:
            lines.append("3. АНАЛІЗ ЗА КАТЕГОРІЯМИ ТОВАРІВ:")
            lines.append(self._category_stats.to_string(index=False))
            lines.append("")

        if self._corr_matrix is not None:
            lines.append("4. МАТРИЦЯ КОРЕЛЯЦІЙ ПІРСОНА:")
            lines.append(self._corr_matrix.to_string())
            lines.append("")

        if self._stat_tests:
            lines.append("5. СТАТИСТИЧНІ ТЕСТИ ТА ЛІНІЙНА РЕГРЕСІЯ (SciPy):")
            lines.append(
                f"   - Критерій Шапіро–Уїлка (net_revenue): W = {self._stat_tests['shapiro_stat']}, "
                f"p = {self._stat_tests['shapiro_p']} "
                f"({'розподіл близький до нормального' if self._stat_tests['is_normal_alpha_05'] else 'відхилення від нормальності'})"
            )
            lines.append(
                f"   - Кореляція (ad_spend vs net_revenue): r = {self._stat_tests['pearson_r']}, "
                f"p-value = {self._stat_tests['p_value']}"
            )
            lines.append(
                f"   - Рівняння лінійної регресії: y = {self._stat_tests['slope']} * x "
                f"+ {self._stat_tests['intercept']}"
            )
            lines.append(f"   - Коефіцієнт детермінації R^2: {self._stat_tests['r_squared']}")
            lines.append("")

        lines.append("=" * 78)

        with open(out, "w", encoding="utf-8") as file:
            file.write("\n".join(lines))

        return str(out)

# КЛАС 3: РОЗШИРЕНА ВІЗУАЛІЗАЦІЯ З АВТОМАТИЧНИМ ЗБЕРЕЖЕННЯМ
class DataVisualizer:
    """
    Клас відповідає за побудову комплексу статистичних діаграм різного типу
    засобами Matplotlib і Seaborn та їх автоматичне збереження у PNG-файли.
    """

    def __init__(self, df: pd.DataFrame, output_dir: str = "output/charts") -> None:
        if df is None or df.empty:
            raise ValueError("Неможливо побудувати графіки для порожнього набору даних.")
        self._df: pd.DataFrame = df.copy()
        self._output_dir: Path = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._saved_files: list[str] = []

        # Єдина візуальна тема для всіх графіків
        sns.set_theme(style="whitegrid", palette="muted")

    @property
    def saved_files(self) -> list[str]:
        """Повертає список шляхів до збережених графіків."""
        return list(self._saved_files)

    def plot_revenue_distribution(self) -> str:
        """
        Графік 1: Гістограма розподілу чистої виручки з оцінкою щільності
        та вертикальними лініями середнього і медіанного значень.
        """
        fig, ax = plt.subplots(figsize=(9, 5))
        sns.histplot(
            data=self._df,
            x="net_revenue",
            bins=18,
            kde=True,
            color="#2b6cb0",
            edgecolor="white",
            alpha=0.75,
            ax=ax,
        )

        mean_val = float(self._df["net_revenue"].mean())
        median_val = float(self._df["net_revenue"].median())

        ax.axvline(mean_val, color="#e53e3e", linestyle="--", linewidth=2, label=f"Середнє: {mean_val:,.0f} грн")
        ax.axvline(median_val, color="#2f855a", linestyle="-.", linewidth=2, label=f"Медіана: {median_val:,.0f} грн")

        ax.set_title("Розподіл чистої виручки за операціями", fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Чиста виручка, грн", fontsize=11)
        ax.set_ylabel("Кількість операцій", fontsize=11)
        ax.grid(True, linestyle=":", alpha=0.7)
        ax.legend()

        file_path = self._output_dir / "01_revenue_distribution.png"
        fig.tight_layout()
        fig.savefig(file_path, dpi=150)
        plt.close(fig)
        return str(file_path)

    def plot_category_revenue(self, category_df: pd.DataFrame) -> str:
        """
        Графік 2: Стовпчаста діаграма сумарної чистої виручки за категоріями товарів
        """
        fig, ax = plt.subplots(figsize=(9, 5))
        sns.barplot(
            data=category_df,
            x="category",
            y="total_net_revenue",
            hue="category",
            palette="viridis",
            legend=False,
            ax=ax,
        )

        # Додаємо числові підписи над кожним стовпцем
        for container in ax.containers:
            ax.bar_label(container, fmt="%.0f", padding=4, fontsize=9)

        ax.set_title("Сумарна чиста виручка за категоріями товарів", fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Категорія товару", fontsize=11)
        ax.set_ylabel("Сумарна чиста виручка, грн", fontsize=11)
        ax.grid(True, axis="y", linestyle=":", alpha=0.7)

        file_path = self._output_dir / "02_category_revenue.png"
        fig.tight_layout()
        fig.savefig(file_path, dpi=150)
        plt.close(fig)
        return str(file_path)

    def plot_correlation_heatmap(self, corr_matrix: pd.DataFrame) -> str:
        """
        Графік 3: Теплова карта кореляційної матриці числових показників
        """
        fig, ax = plt.subplots(figsize=(8, 6.5))
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            vmin=-1.0,
            vmax=1.0,
            linewidths=0.5,
            square=True,
            cbar_kws={"label": "Коефіцієнт кореляції Пірсона (r)"},
            ax=ax,
        )
        ax.set_title("Матриця кореляцій показників продажів", fontsize=13, fontweight="bold", pad=12)

        file_path = self._output_dir / "03_correlation_heatmap.png"
        fig.tight_layout()
        fig.savefig(file_path, dpi=150)
        plt.close(fig)
        return str(file_path)

    def plot_ad_spend_regression(self, stat_tests: dict[str, Any]) -> str:
        """
        Графік 4: Діаграма розсіювання (ad_spend vs net_revenue) із побудованою
        лінією лінійної регресії та відображенням рівняння й R^2
        """
        fig, ax = plt.subplots(figsize=(9, 5))

        x = self._df["ad_spend"].to_numpy(dtype=float)
        y = self._df["net_revenue"].to_numpy(dtype=float)

        ax.scatter(
            x,
            y,
            color="#3182ce",
            alpha=0.7,
            edgecolors="white",
            s=55,
            label="Спостереження",
        )

        # Побудова лінії тренду на основі коефіцієнтів із SciPy
        x_line = np.linspace(float(np.min(x)), float(np.max(x)), 100)
        y_line = stat_tests["slope"] * x_line + stat_tests["intercept"]
        ax.plot(
            x_line,
            y_line,
            color="#d62728",
            linewidth=2.2,
            label=(
                f"Тренд: y = {stat_tests['slope']:.2f}x + {stat_tests['intercept']:.0f}\n"
                f"(r = {stat_tests['pearson_r']:.3f}, R² = {stat_tests['r_squared']:.3f})"
            ),
        )

        ax.set_title(
            "Залежність чистої виручки від витрат на рекламу",
            fontsize=13,
            fontweight="bold",
            pad=12,
        )
        ax.set_xlabel("Витрати на рекламу (ad_spend), грн", fontsize=11)
        ax.set_ylabel("Чиста виручка (net_revenue), грн", fontsize=11)
        ax.grid(True, linestyle=":", alpha=0.7)
        ax.legend(loc="upper left")

        file_path = self._output_dir / "04_ad_spend_regression.png"
        fig.tight_layout()
        fig.savefig(file_path, dpi=150)
        plt.close(fig)
        return str(file_path)

    def generate_all_charts(
        self,
        category_df: pd.DataFrame,
        corr_matrix: pd.DataFrame,
        stat_tests: dict[str, Any],
    ) -> list[str]:
        """
        Послідовно генерує та зберігає всі 4 діаграми.
        """
        self._saved_files = [
            self.plot_revenue_distribution(),
            self.plot_category_revenue(category_df),
            self.plot_correlation_heatmap(corr_matrix),
            self.plot_ad_spend_regression(stat_tests),
        ]
        return self.saved_files

# ДОПОМІЖНІ ФУНКЦІЇ ІНТЕРФЕЙСУ ТА ГОЛОВНИЙ ЦИКЛ ПРОГРАМИ
def print_menu() -> None:
    """Виводить головне меню діалогу з користувачем."""
    print("\n" + "=" * 64)
    print("   СИСТЕМА КОМПЛЕКСНОГО АНАЛІЗУ ДАНИХ ПРОДАЖІВ (PyData ООП)")
    print("=" * 64)
    print("  1. Згенерувати тестовий файл даних (CSV / JSON / Excel)")
    print("  2. Завантажити дані з файлу (CSV / JSON / Excel)")
    print("  3. Виконати очищення та попереднє оброблення даних")
    print("  4. Провести статистичний аналіз (Pandas, NumPy, SciPy)")
    print("  5. Побудувати та зберегти графіки (Matplotlib, Seaborn)")
    print("  6. Експортувати очищені дані (CSV/JSON/Excel) та звіт (TXT)")
    print("  7. Запустити повний автоматичний цикл (усі етапи одразу)")
    print("  0. Вихід із програми")
    print("=" * 64)


def get_float_input(prompt: str, default: float, min_val: float = 0.5) -> float:
    """Безпечно зчитує додатне дійсне число з консолі."""
    raw = input(f"{prompt} [за замовчуванням {default}]: ").strip()
    if not raw:
        return default
    try:
        value = float(raw.replace(",", "."))
        if value < min_val:
            raise ValueError(f"Значення має бути не менше {min_val}.")
        return value
    except ValueError as exc:
        print(f"[ПОМИЛКА ВВЕДЕННЯ] {exc} Використано значення за замовчуванням ({default}).")
        return default


def run_full_pipeline(loader: DataLoader, output_dir: str = "output") -> tuple[DataAnalyzer, DataVisualizer]:
    """
    Виконує повний наскрізний конвеєр оброблення:
    генерація CSV -> імпорт -> очищення -> статистика -> 4 графіки -> експорт.
    """
    print("\n[КРОК 1/5] Генерація та імпорт тестового набору даних...")
    sample_path = loader.generate_sample_file(f"{output_dir}/sample_sales.csv", fmt="csv")
    raw_df = loader.load_data(sample_path)
    print(f"   Успішно завантажено {len(raw_df)} записів із файлу '{sample_path}'.")

    print("[КРОК 2/5] Очищення даних (дублікати, пропуски, викиди Z < 3.0)...")
    clean_df = loader.clean_data_pipeline(
        missing_strategy="median", remove_outliers=True, z_threshold=3.0
    )
    rep = loader.cleaning_report
    print(
        f"   Очищення завершено: дублікатів видалено={rep['duplicates_removed']}, "
        f"пропусків оброблено={rep['missing_values_handled']}, "
        f"викидів видалено={rep['outliers_removed']}, "
        f"фінальних записів={rep['final_rows']}."
    )

    print("[КРОК 3/5] Виконання статистичного аналізу (Pandas, NumPy, SciPy)...")
    analyzer = DataAnalyzer(clean_df)
    desc = analyzer.compute_descriptive_stats()
    cat_stats = analyzer.analyze_by_category()
    corr = analyzer.compute_correlations()
    tests = analyzer.run_statistical_tests()

    print("\n--- Описова статистика ---")
    print(desc.to_string(index=False))
    print("\n--- Показники за категоріями ---")
    print(cat_stats.to_string(index=False))
    print(
        f"\n--- Результати SciPy ---\n"
        f"Шапіро–Уїлк (net_revenue): W = {tests['shapiro_stat']}, p = {tests['shapiro_p']}\n"
        f"Лінійна регресія (ad_spend -> net_revenue): "
        f"y = {tests['slope']}x + {tests['intercept']}, r = {tests['pearson_r']}, R² = {tests['r_squared']}"
    )

    print("\n[КРОК 4/5] Побудова та автозбереження 4 діаграм (Matplotlib + Seaborn)...")
    visualizer = DataVisualizer(clean_df, output_dir=f"{output_dir}/charts")
    chart_paths = visualizer.generate_all_charts(cat_stats, corr, tests)
    for path in chart_paths:
        print(f"   Збережено графік: {path}")

    print("[КРОК 5/5] Експорт очищених даних (CSV, JSON, Excel) та текстового звіту...")
    exported_files = loader.export_data(
        output_dir=output_dir, base_name="cleaned_sales", formats=["csv", "json", "xlsx"]
    )
    report_path = analyzer.save_text_report(
        output_path=f"{output_dir}/analysis_report.txt", cleaning_report=rep
    )
    for path in exported_files + [report_path]:
        print(f"   Збережено файл:   {path}")

    print("\n[ГОТОВО] Повний цикл комплексного аналізу успішно завершено.")
    return analyzer, visualizer


def main() -> None:
    """Точка входу програми з інтерактивним меню та обробленням виключень."""
    loader = DataLoader()
    analyzer: Optional[DataAnalyzer] = None
    visualizer: Optional[DataVisualizer] = None

    while True:
        print_menu()
        try:
            choice = input("Оберіть дію (0-7): ").strip()

            if choice == "1":
                fmt = input("Формат тестового файлу (csv / json / xlsx) [csv]: ").strip().lower() or "csv"
                default_name = f"output/sample_sales.{fmt}"
                filepath = input(f"Шлях збереження [{default_name}]: ").strip() or default_name
                created = loader.generate_sample_file(filepath, fmt=fmt)
                print(f"[УСПІХ] Тестовий набір даних створено: {created}")

            elif choice == "2":
                filepath = input("Введіть шлях до файлу (.csv, .json, .xlsx): ").strip()
                df = loader.load_data(filepath)
                analyzer = None
                visualizer = None
                print(f"[УСПІХ] Завантажено {len(df)} рядків і {len(df.columns)} стовпців.")
                print("Перші 5 записів:\n", df.head().to_string(index=False))

            elif choice == "3":
                strategy = (
                    input("Стратегія для пропусків (median / mean / drop) [median]: ")
                    .strip()
                    .lower()
                    or "median"
                )
                rem_out_str = input("Видаляти статистичні викиди? (y/n) [y]: ").strip().lower()
                remove_outliers = rem_out_str != "n"
                z_val = get_float_input("Поріг Z-оцінки", default=3.0, min_val=1.0) if remove_outliers else 3.0

                clean_df = loader.clean_data_pipeline(
                    missing_strategy=strategy,
                    remove_outliers=remove_outliers,
                    z_threshold=z_val,
                )
                analyzer = None
                visualizer = None
                print("\n[УСПІХ] Очищення виконано. Звіт:")
                for key, val in loader.cleaning_report.items():
                    print(f"  - {key}: {val}")
                print(f"Додано розрахункові поля: revenue, net_revenue. Рядків: {len(clean_df)}.")

            elif choice == "4":
                clean_df = loader.clean_data
                if clean_df is None:
                    raise RuntimeError("Спочатку завантажте (п. 2) та очистіть (п. 3) дані.")
                analyzer = DataAnalyzer(clean_df)
                print("\n--- 1. Описова статистика ---")
                print(analyzer.compute_descriptive_stats().to_string(index=False))
                print("\n--- 2. Аналіз за категоріями ---")
                print(analyzer.analyze_by_category().to_string(index=False))
                print("\n--- 3. Кореляційна матриця ---")
                print(analyzer.compute_correlations().to_string())
                tests = analyzer.run_statistical_tests()
                print("\n--- 4. Статистичні тести (SciPy) ---")
                print(f"  Шапіро–Уїлк: W = {tests['shapiro_stat']}, p = {tests['shapiro_p']}")
                print(
                    f"  Регресія (ad_spend -> net_revenue): y = {tests['slope']}x + {tests['intercept']}, "
                    f"r = {tests['pearson_r']}, R² = {tests['r_squared']} (p = {tests['p_value']})"
                )

            elif choice == "5":
                if analyzer is None or analyzer.category_stats is None or analyzer.corr_matrix is None:
                    raise RuntimeError("Спочатку виконайте статистичний аналіз (п. 4).")
                clean_df = loader.clean_data
                assert clean_df is not None
                visualizer = DataVisualizer(clean_df, output_dir="output/charts")
                paths = visualizer.generate_all_charts(
                    analyzer.category_stats, analyzer.corr_matrix, analyzer.stat_tests
                )
                print("[УСПІХ] Згенеровано та збережено графіки:")
                for p in paths:
                    print(f"  - {p}")

            elif choice == "6":
                if loader.clean_data is None:
                    raise RuntimeError("Немає очищених даних для експорту (виконайте п. 3).")
                raw_fmts = input("Формати експорту через кому (csv, json, xlsx) [csv,json,xlsx]: ").strip()
                formats = [f.strip() for f in raw_fmts.split(",") if f.strip()] if raw_fmts else ["csv", "json", "xlsx"]

                saved = loader.export_data(output_dir="output", base_name="cleaned_sales", formats=formats)
                for p in saved:
                    print(f"[УСПІХ] Експортовано дані: {p}")

                if analyzer is not None:
                    rep_file = analyzer.save_text_report(
                        output_path="output/analysis_report.txt",
                        cleaning_report=loader.cleaning_report,
                    )
                    print(f"[УСПІХ] Збережено текстовий звіт: {rep_file}")
                else:
                    print("[ІНФО] Текстовий звіт не створено, оскільки статистичний аналіз (п. 4) ще не запускався.")

            elif choice == "7":
                analyzer, visualizer = run_full_pipeline(loader, output_dir="output")

            elif choice == "0":
                print("Роботу програми завершено. До побачення!")
                break

            else:
                print("[ПОМИЛКА] Некоректний пункт меню. Введіть число від 0 до 7.")

        except (FileNotFoundError, ValueError, KeyError, RuntimeError, ImportError, PermissionError) as err:
            print(f"\n[ПОМИЛКА ВИКОНАННЯ] {err}")
        except KeyboardInterrupt:
            print("\nРоботу програми перервано користувачем.")
            break
        except Exception as unexpected_err:
            print(f"\n[НЕПЕРЕДБАЧЕНА ПОМИЛКА] {type(unexpected_err).__name__}: {unexpected_err}")


if __name__ == "__main__":
    main()