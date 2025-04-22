import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

# Настройка стиля графиков
plt.style.use('seaborn')
sns.set_palette("husl")

def print_math_results(title, results):
    print(f"\n{title}")
    print("=" * 50)
    for key, value in results.items():
        print(f"{key}: {value:.6f}")
    print("=" * 50)

# Задание 1.1: Критерий знаков
def task_1_1():
    np.random.seed(42)
    n = 500
    X = np.random.normal(2, np.sqrt(5), n)
    Y = np.random.normal(2, np.sqrt(8), n)
    
    # Вычисление статистики критерия знаков
    signs = np.sign(X - Y)
    w = min(np.sum(signs == -1), np.sum(signs == 1))
    t_hat = (w - n/2) / np.sqrt(n/4)
    p_value = 2 * (1 - stats.norm.cdf(abs(t_hat)))
    
    # Математические выкладки
    results = {
        "Число знаков '-':": np.sum(signs == -1),
        "Число знаков '+':": np.sum(signs == 1),
        "Минимальное число знаков (w):": w,
        "Статистика t_hat:": t_hat,
        "p-value:": p_value,
        "Критическое значение (α=0.05):": stats.norm.ppf(0.975)
    }
    
    print_math_results("Задание 1.1: Критерий знаков", results)
    
    # Визуализация
    plt.figure(figsize=(12, 6))
    plt.hist(X, alpha=0.5, label='X ~ N(2,5)', bins=30, density=True)
    plt.hist(Y, alpha=0.5, label='Y ~ N(2,8)', bins=30, density=True)
    x = np.linspace(min(min(X), min(Y)), max(max(X), max(Y)), 1000)
    plt.plot(x, stats.norm.pdf(x, 2, np.sqrt(5)), 'b-', alpha=0.7)
    plt.plot(x, stats.norm.pdf(x, 2, np.sqrt(8)), 'r-', alpha=0.7)
    plt.legend()
    plt.title('Распределения выборок X и Y')
    plt.savefig('task_1_1.png', dpi=300, bbox_inches='tight')
    plt.close()

# Задание 1.2: Критерий знаков со сдвигом
def task_1_2():
    np.random.seed(42)
    n = 500
    X = np.random.normal(2, np.sqrt(5), n)
    Y = np.random.normal(2.2, np.sqrt(7), n)
    
    # Вычисление статистики критерия знаков
    signs = np.sign(X - Y)
    w = min(np.sum(signs == -1), np.sum(signs == 1))
    t_hat = (w - n/2) / np.sqrt(n/4)
    p_value = 2 * (1 - stats.norm.cdf(abs(t_hat)))
    
    # Математические выкладки
    results = {
        "Число знаков '-':": np.sum(signs == -1),
        "Число знаков '+':": np.sum(signs == 1),
        "Минимальное число знаков (w):": w,
        "Статистика t_hat:": t_hat,
        "p-value:": p_value,
        "Критическое значение (α=0.05):": stats.norm.ppf(0.975)
    }
    
    print_math_results("Задание 1.2: Критерий знаков со сдвигом", results)
    
    # Визуализация
    plt.figure(figsize=(12, 6))
    plt.hist(X, alpha=0.5, label='X ~ N(2,5)', bins=30, density=True)
    plt.hist(Y, alpha=0.5, label='Y ~ N(2.2,7)', bins=30, density=True)
    x = np.linspace(min(min(X), min(Y)), max(max(X), max(Y)), 1000)
    plt.plot(x, stats.norm.pdf(x, 2, np.sqrt(5)), 'b-', alpha=0.7)
    plt.plot(x, stats.norm.pdf(x, 2.2, np.sqrt(7)), 'r-', alpha=0.7)
    plt.legend()
    plt.title('Распределения выборок X и Y со сдвигом')
    plt.savefig('task_1_2.png', dpi=300, bbox_inches='tight')
    plt.close()

# Задание 2.1: Ранговый коэффициент корреляции Кендалла
def task_2_1():
    np.random.seed(42)
    n = 500
    X = np.random.normal(2, np.sqrt(5), n)
    Y = 3 * X - 5 + np.random.normal(0, 1, n)
    
    # Вычисление коэффициента Кендалла
    tau, p_value = stats.kendalltau(X, Y)
    
    # Математические выкладки
    results = {
        "Коэффициент Кендалла (τ):": tau,
        "p-value:": p_value,
        "Критическое значение (α=0.05):": stats.norm.ppf(0.975),
        "Стандартная ошибка:": np.sqrt(2*(2*n+5)/(9*n*(n-1)))
    }
    
    print_math_results("Задание 2.1: Ранговый коэффициент корреляции Кендалла", results)
    
    # Визуализация
    plt.figure(figsize=(12, 6))
    plt.scatter(X, Y, alpha=0.5)
    plt.title('Зависимость Y от X')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.savefig('task_2_1.png', dpi=300, bbox_inches='tight')
    plt.close()

# Задание 2.2: Случай слабой зависимости
def task_2_2():
    np.random.seed(42)
    n = 500
    X = np.random.normal(2, np.sqrt(5), n)
    Y = 0.05 * X - 5 + np.random.normal(0, 1, n)
    
    # Вычисление коэффициента Кендалла
    tau, p_value = stats.kendalltau(X, Y)
    
    # Математические выкладки
    results = {
        "Коэффициент Кендалла (τ):": tau,
        "p-value:": p_value,
        "Критическое значение (α=0.05):": stats.norm.ppf(0.975),
        "Стандартная ошибка:": np.sqrt(2*(2*n+5)/(9*n*(n-1)))
    }
    
    print_math_results("Задание 2.2: Случай слабой зависимости", results)
    
    # Визуализация
    plt.figure(figsize=(12, 6))
    plt.scatter(X, Y, alpha=0.5)
    plt.title('Слабая зависимость Y от X')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.savefig('task_2_2.png', dpi=300, bbox_inches='tight')
    plt.close()

# Задание 3.1: Критерий отсутствия автокорреляции
def task_3_1():
    np.random.seed(42)
    n = 500
    X = np.random.normal(3, np.sqrt(7), n)
    
    # Вычисление коэффициента автокорреляции
    numerator = n * np.sum(X[:-1] * X[1:]) - np.sum(X)**2 + n * X[0] * X[-1]
    denominator = n * np.sum(X**2) - np.sum(X)**2
    rho = numerator / denominator
    
    # Вычисление статистики
    M_rho = -1 / (n - 1)
    D_rho = (n * (n - 3)) / ((n + 1) * (n - 1)**2)
    t_hat = (rho - M_rho) / np.sqrt(D_rho)
    p_value = 2 * (1 - stats.norm.cdf(abs(t_hat)))
    
    # Математические выкладки
    results = {
        "Коэффициент автокорреляции (ρ):": rho,
        "Математическое ожидание (M[ρ]):": M_rho,
        "Дисперсия (D[ρ]):": D_rho,
        "Статистика t_hat:": t_hat,
        "p-value:": p_value,
        "Критическое значение (α=0.05):": stats.norm.ppf(0.975)
    }
    
    print_math_results("Задание 3.1: Критерий отсутствия автокорреляции", results)
    
    # Визуализация
    plt.figure(figsize=(12, 6))
    plt.plot(X)
    plt.title('Временной ряд X')
    plt.xlabel('Индекс')
    plt.ylabel('Значение')
    plt.savefig('task_3_1.png', dpi=300, bbox_inches='tight')
    plt.close()

# Задание 3.2: Модифицированная выборка
def task_3_2():
    np.random.seed(42)
    n = 500
    X = np.random.normal(3, np.sqrt(7), n)
    X_mod = np.zeros_like(X)
    X_mod[0] = X[0]
    
    # Исправленная формула для избежания переполнения
    for i in range(1, n):
        X_mod[i] = 0.5 * X[i] - 0.1 * X_mod[i-1] + 0.1 * np.random.normal(0, 1)
    
    # Вычисление коэффициента автокорреляции
    numerator = n * np.sum(X_mod[:-1] * X_mod[1:]) - np.sum(X_mod)**2 + n * X_mod[0] * X_mod[-1]
    denominator = n * np.sum(X_mod**2) - np.sum(X_mod)**2
    rho = numerator / denominator
    
    # Вычисление статистики
    M_rho = -1 / (n - 1)
    D_rho = (n * (n - 3)) / ((n + 1) * (n - 1)**2)
    t_hat = (rho - M_rho) / np.sqrt(D_rho)
    p_value = 2 * (1 - stats.norm.cdf(abs(t_hat)))
    
    # Математические выкладки
    results = {
        "Коэффициент автокорреляции (ρ):": rho,
        "Математическое ожидание (M[ρ]):": M_rho,
        "Дисперсия (D[ρ]):": D_rho,
        "Статистика t_hat:": t_hat,
        "p-value:": p_value,
        "Критическое значение (α=0.05):": stats.norm.ppf(0.975)
    }
    
    print_math_results("Задание 3.2: Модифицированная выборка", results)
    
    # Визуализация
    plt.figure(figsize=(12, 6))
    plt.plot(X_mod)
    plt.title('Модифицированный временной ряд')
    plt.xlabel('Индекс')
    plt.ylabel('Значение')
    plt.savefig('task_3_2.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    print("Начало выполнения заданий...")
    task_1_1()
    task_1_2()
    task_2_1()
    task_2_2()
    task_3_1()
    task_3_2()
    print("\nВсе задания выполнены!") 