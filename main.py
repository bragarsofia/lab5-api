# Модель: Оптимальне керування процесом очищення водойми
# Брагар Софія, група АІ-233

from flask import Flask, request, jsonify
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

app = Flask(__name__)
# Налаштування для коректного відображення кирилиці в JSON
app.config['JSON_AS_ASCII'] = False

class WaterOptimizer:
    def __init__(self, V=1000, Q=50, k=0.1, C_initial=10, C_target=2, Cin_max=5):
        self.V = V           # Об'єм водойми
        self.Q = Q           # Потік води
        self.k = k           # Коефіцієнт самоочищення
        self.C_initial = C_initial  # Початкова концентрація
        self.C_target = C_target    # Цільова концентрація
        self.Cin_max = Cin_max      # Максимально допустиме керування

    def cstr_ode(self, t, C, Cin_profile, t_points):
        # Інтерполяція керування u(t)
        u_t = np.interp(t, t_points, Cin_profile)
        dCdt = (self.Q / self.V) * u_t - (self.Q / self.V + self.k) * C
        return dCdt

    def objective_minimum_effort(self, Cin_profile, T_fix, N_steps):
        t_points = np.linspace(0, T_fix, N_steps)
        # Розв'язання диференціального рівняння
        sol = solve_ivp(self.cstr_ode, (0, T_fix), [self.C_initial],
                        args=(Cin_profile, t_points), method='RK45', t_eval=t_points)
        
        C_final = sol.y[0, -1]
        # Штраф за недосягнення цілі
        penalty = 0 if C_final <= self.C_target else (C_final - self.C_target)**2 * 1e5
        
        dt = T_fix / (N_steps - 1)
        # Функціонал витрат (мінімізація зусиль)
        return np.sum(Cin_profile**2) * dt + penalty

@app.route('/calculate', methods=['POST'])
def calculate():
    data = request.get_json()
    
    # Отримуємо параметр T_fix з JSON (за замовчуванням 10)
    t_fix = float(data.get('t_fix', 10))
    
    # Ініціалізація моделі з конкретними параметрами для обчислення
    optimizer = WaterOptimizer(
        V=1000, 
        Q=50, 
        k=0.1, 
        C_initial=10, 
        C_target=2, 
        Cin_max=5
    )
    
    # Початкове наближення для оптимізатора
    initial_guess = np.ones(20) * (optimizer.Cin_max / 2)
    
    # Виконання оптимізації
    res = minimize(
        optimizer.objective_minimum_effort, 
        initial_guess, 
        args=(t_fix, 20), 
        method='SLSQP',
        bounds=[(0, optimizer.Cin_max)] * 20
    )
    
    return jsonify({
        "model": "Оптимальне керування процесом очищення водойми",
        "author": "Брагар Софія, АІ-233",
        "input_t_fix": t_fix,
        "optimal_cost": round(float(res.fun), 4),
        "avg_control": round(float(np.mean(res.x)), 4),
        "status": "Розрахунок завершено успішно"
    })

if __name__ == '__main__':
    # Запуск на порту 5000
    app.run(host='0.0.0.0', port=5000)
