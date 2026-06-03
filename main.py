import numpy as np
import scipy.io as sio
from scipy.optimize import minimize


def _huber(r, k):
    """Huber loss: 잔차가 작으면 L2(부드러움), 크면 L1(robust)."""
    abs_r = np.abs(r)
    return np.where(abs_r <= k, 0.5 * r ** 2, k * (abs_r - 0.5 * k))


def _huber_grad(r, k):
    """Huber loss의 잔차에 대한 그래디언트."""
    return np.where(np.abs(r) <= k, r, k * np.sign(r))


def _solve_step(d_hat_u, p_bs, x_init, delta, weights, huber_k=2.0):
    """가중 Huber 손실 + 부등식 제약을 SLSQP로 한 스텝 풀이."""
    n_bs = p_bs.shape[1]

    def obj(xv):
        r_hat = np.linalg.norm(p_bs - xv[:, None], axis=0)
        res = d_hat_u - r_hat
        return float(np.sum(weights * _huber(res, huber_k)))

    def obj_grad(xv):
        diff = p_bs - xv[:, None]
        r_hat = np.linalg.norm(diff, axis=0)
        res = d_hat_u - r_hat
        hg = _huber_grad(res, huber_k)
        # d Huber / d x = w * Huber'(res) * d(d_hat - r_hat)/dx
        # d r_hat / dx = -(p_bs - x) / r_hat 이므로 d res / dx = (p_bs - x) / r_hat
        return (weights * hg * diff / (r_hat + 1e-9)).sum(axis=1)

    def make_constraint(i):
        def c(xv):
            return d_hat_u[i] + delta - np.linalg.norm(p_bs[:, i] - xv)
        def c_jac(xv):
            diff = p_bs[:, i] - xv
            r = np.linalg.norm(diff)
            return diff / (r + 1e-9)
        return {'type': 'ineq', 'fun': c, 'jac': c_jac}

    constraints = [make_constraint(i) for i in range(n_bs)]
    bounds = [(-80.0, 80.0), (-50.0, 50.0)]

    try:
        res = minimize(obj, x_init, jac=obj_grad,
                       method='SLSQP', constraints=constraints, bounds=bounds,
                       options={'maxiter': 100, 'ftol': 1e-7})
        x = res.x.copy()
    except Exception:
        x = x_init.copy()

    x[0] = float(np.clip(x[0], -80.0, 80.0))
    x[1] = float(np.clip(x[1], -50.0, 50.0))
    return x


def _iterative_constrained_localize(d_hat_u, p_bs,
                                    delta_in=0.7, delta_out=1.0,
                                    decay=1.5, huber_k=2.0, n_outer=5):
    """
    Iterative Reweighted Constrained Huber Localization으로 위치를 추정한다.

    1) NLOS 일방향 편향(d_hat은 거의 항상 d_true 이상)을 부등식 제약으로 인코딩.
         ‖p_bs_i - x‖ <= d_hat_i + delta
    2) 부등식 제약 하에 가중 Huber LS를 풀어 위치 추정 x를 얻는다.
       Huber는 큰 잔차에 대해 L1처럼 작동하여 NLOS 외란에 강건하다.
    3) 첫 풀이의 추정 x가 기지국 격자 [-50,50] × [-20,20] 외부이면 외삽 사용자로
       판정하여 더 큰 delta_out을 사용한다. 외부 사용자는 LOS BS가 적어 잡음이
       크고 부등식이 빡빡할수록 SLSQP가 feasible 해를 찾기 어렵기 때문이다.
    4) 잔차 r_i = d_hat_i - ‖p_bs_i - x‖를 보고 가중치를 갱신:
       w_i = exp(-max(r_i, 0) / decay).
       r_i가 큰 양수인 BS는 NLOS로 부풀려졌을 가능성이 높아 가중치 ↓.
    5) 새 가중치로 다시 풀이. 4-5단계를 n_outer번 반복.

    KKT active set이 LOS BS를 식별하는 효과를 가지므로 그 정보를 다음 반복의
    가중치로 부드럽게 전달하여 추정을 정제한다.
    """
    n_bs = p_bs.shape[1]

    # 초기점: 1/d_hat^2 가중평균
    w_init = 1.0 / (d_hat_u ** 2 + 1e-3)
    x = (p_bs * w_init).sum(axis=1) / w_init.sum()

    # 1차 풀이 (균등 가중치)
    weights = np.ones(n_bs)
    x = _solve_step(d_hat_u, p_bs, x, delta_in, weights, huber_k)

    # 외삽 사용자 판정
    is_outside = (abs(x[0]) > 50.0) or (abs(x[1]) > 20.0)
    delta = delta_out if is_outside else delta_in

    # 반복적 가중치 갱신
    for _ in range(n_outer - 1):
        r_final = np.linalg.norm(p_bs - x[:, None], axis=0)
        residual = d_hat_u - r_final
        weights = np.exp(-np.maximum(residual, 0.0) / decay)
        x = _solve_step(d_hat_u, p_bs, x, delta, weights, huber_k)

    return x


def your_algorithm(d_hat_u, p_bs):
    """
    한 사용자의 18개 RTT 측정값과 18개 기지국 좌표를 받아
    사용자 위치 (x, y)를 추정한다.

    NLOS 일방향 편향을 부등식 제약으로 인코딩한 제약 최적화 문제를
    Huber 손실 + 잔차 기반 가중치 갱신과 결합하여 반복 풀이한다.
    격자 외부 사용자에는 더 큰 tolerance를 적용한다.
    """
    return _iterative_constrained_localize(d_hat_u, p_bs)


def main():
    mat_path = 'DH_FR1.mat'
    data = sio.loadmat(mat_path, squeeze_me=False)

    p_bs_key = 'p_bs' if 'p_bs' in data else 'BS_positions'
    p_bs = np.asarray(data[p_bs_key], dtype=float)
    d_hat = np.asarray(data['d_hat'], dtype=float)

    num_user = d_hat.shape[1]
    p_hat = np.zeros((2, num_user))
    for u in range(num_user):
        p_hat[:, u] = your_algorithm(d_hat[:, u], p_bs)

    return p_hat


if __name__ == "__main__":
    p_hat = main()
    print(f"Done. p_hat shape = {p_hat.shape}")
