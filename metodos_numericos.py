"""
=========================================================================
ANÁLISIS NUMÉRICO - MÉTODOS PARA ENCONTRAR RAÍCES
Métodos: Bisección, Newton-Raphson, Secante
=========================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Tuple, List
import warnings
warnings.filterwarnings('ignore')

# =========================================================================
# DEFINICIÓN DE FUNCIONES Y SUS DERIVADAS
# =========================================================================

# Ecuación 1: x^3 - e^(-0.8x) = 20
def f1(x):
    return x**3 - np.exp(-0.8*x) - 20

def df1(x):
    return 3*x**2 + 0.8*np.exp(-0.8*x)

# Ecuación 2: 3*sin(0.5x) - 0.5x + 2 = 0
def f2(x):
    return 3*np.sin(0.5*x) - 0.5*x + 2

def df2(x):
    return 1.5*np.cos(0.5*x) - 0.5

# Ecuación 3: x^3 - x^2*e^(-0.5x) - 3x = -1
def f3(x):
    return x**3 - x**2*np.exp(-0.5*x) - 3*x + 1

def df3(x):
    return 3*x**2 - 2*x*np.exp(-0.5*x) + 0.5*x**2*np.exp(-0.5*x) - 3

# Ecuación 4: cos^2(x) - 0.5x*e^(0.3x) + 5 = 0
def f4(x):
    return np.cos(x)**2 - 0.5*x*np.exp(0.3*x) + 5

def df4(x):
    return -2*np.cos(x)*np.sin(x) - 0.5*np.exp(0.3*x) - 0.15*x*np.exp(0.3*x)

# =========================================================================
# MÉTODO DE BISECCIÓN
# =========================================================================
def metodo_biseccion(f, a, b, tol=1e-6, max_iter=100):
    """
    Implementa el método de bisección para encontrar raíces.
    
    Retorna:
        raiz: valor de la raíz encontrada
        df: DataFrame con las iteraciones
        errores: lista de errores en cada iteración
    """
    iteraciones = []
    errores = []
    
    for i in range(1, max_iter + 1):
        c = (a + b) / 2
        fc = f(c)
        fa = f(a)
        error = abs(b - a)
        
        iteraciones.append({
            'Iteración': i,
            'a': a,
            'b': b,
            'c': c,
            'f(c)': fc,
            'Error': error
        })
        
        errores.append(error)
        
        if error < tol or abs(fc) < tol:
            break
        
        if fa * fc < 0:
            b = c
        else:
            a = c
    
    df = pd.DataFrame(iteraciones)
    return c, df, errores

# =========================================================================
# MÉTODO DE NEWTON-RAPHSON
# =========================================================================
def metodo_newton(f, df, x0, tol=1e-6, max_iter=100):
    """
    Implementa el método de Newton-Raphson para encontrar raíces.
    
    Retorna:
        raiz: valor de la raíz encontrada
        df_result: DataFrame con las iteraciones
        errores: lista de errores en cada iteración
    """
    iteraciones = []
    errores = []
    x = x0
    
    for i in range(1, max_iter + 1):
        fx = f(x)
        dfx = df(x)
        
        if abs(dfx) < 1e-10:
            print(f"Advertencia: Derivada muy pequeña en iteración {i}")
            break
        
        x_nuevo = x - fx/dfx
        error = abs(x_nuevo - x)
        
        iteraciones.append({
            'Iteración': i,
            'x_n': x,
            'f(x_n)': fx,
            "f'(x_n)": dfx,
            'x_n+1': x_nuevo,
            'Error': error
        })
        
        errores.append(error)
        
        if error < tol or abs(fx) < tol:
            break
        
        x = x_nuevo
    
    df_result = pd.DataFrame(iteraciones)
    return x_nuevo, df_result, errores

# =========================================================================
# MÉTODO DE LA SECANTE
# =========================================================================
def metodo_secante(f, x0, x1, tol=1e-6, max_iter=100):
    """
    Implementa el método de la secante para encontrar raíces.
    
    Retorna:
        raiz: valor de la raíz encontrada
        df: DataFrame con las iteraciones
        errores: lista de errores en cada iteración
    """
    iteraciones = []
    errores = []
    
    for i in range(1, max_iter + 1):
        fx0 = f(x0)
        fx1 = f(x1)
        
        if abs(fx1 - fx0) < 1e-10:
            print(f"Advertencia: Denominador muy pequeño en iteración {i}")
            break
        
        x2 = x1 - fx1 * (x1 - x0) / (fx1 - fx0)
        error = abs(x2 - x1)
        
        iteraciones.append({
            'Iteración': i,
            'x_n-1': x0,
            'x_n': x1,
            'f(x_n)': fx1,
            'x_n+1': x2,
            'Error': error
        })
        
        errores.append(error)
        
        if error < tol or abs(fx1) < tol:
            break
        
        x0 = x1
        x1 = x2
    
    df = pd.DataFrame(iteraciones)
    return x2, df, errores

# =========================================================================
# FUNCIÓN PARA GRAFICAR FUNCIONES CON RAÍCES
# =========================================================================
def graficar_funcion(f, x_range, raices, titulo, nombre_archivo=None):
    """Grafica una función y marca las raíces encontradas por cada método"""
    plt.figure(figsize=(10, 6))
    x = np.linspace(x_range[0], x_range[1], 1000)
    y = f(x)
    
    plt.plot(x, y, 'b-', linewidth=2, label='f(x)')
    plt.axhline(y=0, color='k', linestyle='--', linewidth=1)
    plt.grid(True, alpha=0.3)
    
    # Marcar raíces
    if 'biseccion' in raices:
        for r in raices['biseccion']:
            plt.plot(r, 0, 'ro', markersize=10, label='Bisección')
    if 'newton' in raices:
        for r in raices['newton']:
            plt.plot(r, 0, 'gs', markersize=10, label='Newton-Raphson')
    if 'secante' in raices:
        for r in raices['secante']:
            plt.plot(r, 0, 'md', markersize=10, label='Secante')
    
    # Eliminar duplicados en la leyenda
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='best')
    
    plt.xlabel('x', fontsize=12)
    plt.ylabel('f(x)', fontsize=12)
    plt.title(titulo, fontsize=14, fontweight='bold')
    
    if nombre_archivo:
        plt.savefig(nombre_archivo, dpi=300, bbox_inches='tight')
    
    plt.tight_layout()

# =========================================================================
# FUNCIÓN PARA GRAFICAR CONVERGENCIA
# =========================================================================
def graficar_convergencia(errores_dict, titulo, nombre_archivo=None):
    """Grafica la convergencia de los métodos en escala logarítmica"""
    plt.figure(figsize=(10, 6))
    
    if 'biseccion' in errores_dict:
        plt.semilogy(range(1, len(errores_dict['biseccion'])+1), 
                     errores_dict['biseccion'], 'r-o', 
                     linewidth=2, markersize=6, label='Bisección')
    
    if 'newton' in errores_dict:
        plt.semilogy(range(1, len(errores_dict['newton'])+1), 
                     errores_dict['newton'], 'g-s', 
                     linewidth=2, markersize=6, label='Newton-Raphson')
    
    if 'secante' in errores_dict:
        plt.semilogy(range(1, len(errores_dict['secante'])+1), 
                     errores_dict['secante'], 'm-d', 
                     linewidth=2, markersize=6, label='Secante')
    
    plt.grid(True, alpha=0.3)
    plt.xlabel('Iteración', fontsize=12)
    plt.ylabel('Error absoluto (escala log)', fontsize=12)
    plt.title(titulo, fontsize=14, fontweight='bold')
    plt.legend(loc='best')
    
    if nombre_archivo:
        plt.savefig(nombre_archivo, dpi=300, bbox_inches='tight')
    
    plt.tight_layout()

# =========================================================================
# PROGRAMA PRINCIPAL
# =========================================================================
def main():
    print("\n" + "="*70)
    print(" "*15 + "ANÁLISIS NUMÉRICO - RESULTADOS")
    print("="*70 + "\n")
    
    tol = 1e-6
    max_iter = 100
    
    # =====================================================================
    # ECUACIÓN 1: x^3 - e^(-0.8x) = 20
    # =====================================================================
    print("ECUACIÓN 1: x³ - e^(-0.8x) - 20 = 0")
    print("-" * 70)
    
    r1_bis, df1_bis, err1_bis = metodo_biseccion(f1, 2.5, 3.5, tol, max_iter)
    print(f"Bisección: raíz = {r1_bis:.8f}, iteraciones = {len(df1_bis)}")
    
    r1_new, df1_new, err1_new = metodo_newton(f1, df1, 3, tol, max_iter)
    print(f"Newton-Raphson: raíz = {r1_new:.8f}, iteraciones = {len(df1_new)}")
    
    r1_sec, df1_sec, err1_sec = metodo_secante(f1, 2.5, 3.5, tol, max_iter)
    print(f"Secante: raíz = {r1_sec:.8f}, iteraciones = {len(df1_sec)}\n")
    
    # Guardar tablas
    df1_bis.to_csv('ecuacion1_biseccion.csv', index=False)
    df1_new.to_csv('ecuacion1_newton.csv', index=False)
    df1_sec.to_csv('ecuacion1_secante.csv', index=False)
    
    # =====================================================================
    # ECUACIÓN 2: 3*sin(0.5x) - 0.5x + 2 = 0
    # =====================================================================
    print("ECUACIÓN 2: 3sin(0.5x) - 0.5x + 2 = 0")
    print("-" * 70)
    
    r2_bis, df2_bis, err2_bis = metodo_biseccion(f2, -10, 0, tol, max_iter)
    print(f"Bisección: raíz = {r2_bis:.8f}, iteraciones = {len(df2_bis)}")
    
    r2_new, df2_new, err2_new = metodo_newton(f2, df2, -5, tol, max_iter)
    print(f"Newton-Raphson: raíz = {r2_new:.8f}, iteraciones = {len(df2_new)}")
    
    r2_sec, df2_sec, err2_sec = metodo_secante(f2, -10, 0, tol, max_iter)
    print(f"Secante: raíz = {r2_sec:.8f}, iteraciones = {len(df2_sec)}\n")
    
    # Guardar tablas
    df2_bis.to_csv('ecuacion2_biseccion.csv', index=False)
    df2_new.to_csv('ecuacion2_newton.csv', index=False)
    df2_sec.to_csv('ecuacion2_secante.csv', index=False)
    
    # =====================================================================
    # ECUACIÓN 3: x^3 - x^2*e^(-0.5x) - 3x = -1 (3 raíces)
    # =====================================================================
    print("ECUACIÓN 3: x³ - x²e^(-0.5x) - 3x + 1 = 0")
    print("-" * 70)
    
    # Raíz 1
    print("Raíz 1:")
    r3_1_bis, df3_1_bis, err3_1_bis = metodo_biseccion(f3, -1, 0, tol, max_iter)
    print(f"  Bisección: raíz = {r3_1_bis:.8f}, iteraciones = {len(df3_1_bis)}")
    
    r3_1_new, df3_1_new, err3_1_new = metodo_newton(f3, df3, -0.5, tol, max_iter)
    print(f"  Newton-Raphson: raíz = {r3_1_new:.8f}, iteraciones = {len(df3_1_new)}")
    
    r3_1_sec, df3_1_sec, err3_1_sec = metodo_secante(f3, -1, 0, tol, max_iter)
    print(f"  Secante: raíz = {r3_1_sec:.8f}, iteraciones = {len(df3_1_sec)}\n")
    
    # Raíz 2
    print("Raíz 2:")
    r3_2_bis, df3_2_bis, err3_2_bis = metodo_biseccion(f3, 0, 1, tol, max_iter)
    print(f"  Bisección: raíz = {r3_2_bis:.8f}, iteraciones = {len(df3_2_bis)}")
    
    r3_2_new, df3_2_new, err3_2_new = metodo_newton(f3, df3, 0.5, tol, max_iter)
    print(f"  Newton-Raphson: raíz = {r3_2_new:.8f}, iteraciones = {len(df3_2_new)}")
    
    r3_2_sec, df3_2_sec, err3_2_sec = metodo_secante(f3, 0, 1, tol, max_iter)
    print(f"  Secante: raíz = {r3_2_sec:.8f}, iteraciones = {len(df3_2_sec)}\n")
    
    # Raíz 3
    print("Raíz 3:")
    r3_3_bis, df3_3_bis, err3_3_bis = metodo_biseccion(f3, 2.5, 3.5, tol, max_iter)
    print(f"  Bisección: raíz = {r3_3_bis:.8f}, iteraciones = {len(df3_3_bis)}")
    
    r3_3_new, df3_3_new, err3_3_new = metodo_newton(f3, df3, 3, tol, max_iter)
    print(f"  Newton-Raphson: raíz = {r3_3_new:.8f}, iteraciones = {len(df3_3_new)}")
    
    r3_3_sec, df3_3_sec, err3_3_sec = metodo_secante(f3, 2.5, 3.5, tol, max_iter)
    print(f"  Secante: raíz = {r3_3_sec:.8f}, iteraciones = {len(df3_3_sec)}\n")
    
    # Guardar tablas
    df3_1_bis.to_csv('ecuacion3_raiz1_biseccion.csv', index=False)
    df3_1_new.to_csv('ecuacion3_raiz1_newton.csv', index=False)
    df3_1_sec.to_csv('ecuacion3_raiz1_secante.csv', index=False)
    
    # =====================================================================
    # ECUACIÓN 4: cos^2(x) - 0.5x*e^(0.3x) + 5 = 0
    # =====================================================================
    print("ECUACIÓN 4: cos²(x) - 0.5xe^(0.3x) + 5 = 0")
    print("-" * 70)
    
    print("Raíz:")
    r4_bis, df4_bis, err4_bis = metodo_biseccion(f4, -8, -6, tol, max_iter)
    print(f"  Bisección: raíz = {r4_bis:.8f}, iteraciones = {len(df4_bis)}")
    
    r4_new, df4_new, err4_new = metodo_newton(f4, df4, -7, tol, max_iter)
    print(f"  Newton-Raphson: raíz = {r4_new:.8f}, iteraciones = {len(df4_new)}")
    
    r4_sec, df4_sec, err4_sec = metodo_secante(f4, -8, -6, tol, max_iter)
    print(f"  Secante: raíz = {r4_sec:.8f}, iteraciones = {len(df4_sec)}\n")
    
    # Guardar tablas
    df4_bis.to_csv('ecuacion4_biseccion.csv', index=False)
    df4_new.to_csv('ecuacion4_newton.csv', index=False)
    df4_sec.to_csv('ecuacion4_secante.csv', index=False)
    
    # =====================================================================
    # GENERACIÓN DE GRÁFICOS
    # =====================================================================
    print("="*70)
    print("GENERACIÓN DE GRÁFICOS")
    print("="*70 + "\n")
    
    # Gráficos de funciones con raíces
    graficar_funcion(f1, [0, 8], 
                    {'biseccion': [r1_bis], 'newton': [r1_new], 'secante': [r1_sec]},
                    'Ecuación 1: x³ - e^(-0.8x) - 20 = 0',
                    'ecuacion1_funcion.png')
    
    graficar_funcion(f2, [-12, 2], 
                    {'biseccion': [r2_bis], 'newton': [r2_new], 'secante': [r2_sec]},
                    'Ecuación 2: 3sin(0.5x) - 0.5x + 2 = 0',
                    'ecuacion2_funcion.png')
    
    graficar_funcion(f3, [-2, 4], 
                    {'biseccion': [r3_1_bis, r3_2_bis, r3_3_bis], 
                     'newton': [r3_1_new, r3_2_new, r3_3_new], 
                     'secante': [r3_1_sec, r3_2_sec, r3_3_sec]},
                    'Ecuación 3: x³ - x²e^(-0.5x) - 3x + 1 = 0',
                    'ecuacion3_funcion.png')
    
    graficar_funcion(f4, [-10, 2], 
                    {'biseccion': [r4_bis], 'newton': [r4_new], 'secante': [r4_sec]},
                    'Ecuación 4: cos²(x) - 0.5xe^(0.3x) + 5 = 0',
                    'ecuacion4_funcion.png')
    
    # Gráficos de convergencia
    graficar_convergencia(
        {'biseccion': err1_bis, 'newton': err1_new, 'secante': err1_sec},
        'Convergencia - Ecuación 1',
        'ecuacion1_convergencia.png')
    
    graficar_convergencia(
        {'biseccion': err2_bis, 'newton': err2_new, 'secante': err2_sec},
        'Convergencia - Ecuación 2',
        'ecuacion2_convergencia.png')
    
    print("✓ Gráficos guardados exitosamente!")
    print("✓ Tablas CSV generadas exitosamente!")
    print("\nArchivos generados:")
    print("  - ecuacion1_biseccion.csv, ecuacion1_newton.csv, ecuacion1_secante.csv")
    print("  - ecuacion2_biseccion.csv, ecuacion2_newton.csv, ecuacion2_secante.csv")
    print("  - ecuacion3_raiz1_biseccion.csv, ecuacion3_raiz1_newton.csv, etc.")
    print("  - ecuacion4_biseccion.csv, ecuacion4_newton.csv, ecuacion4_secante.csv")
    print("  - ecuacion1_funcion.png, ecuacion1_convergencia.png")
    print("  - ecuacion2_funcion.png, ecuacion2_convergencia.png")
    print("  - ecuacion3_funcion.png")
    print("  - ecuacion4_funcion.png")
    
    plt.show()

if __name__ == "__main__":
    main()