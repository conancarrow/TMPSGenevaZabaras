import numpy as np
import torch
from scipy.integrate import solve_ivp

# Lorenz system
def lorenz(t, xyz, sigma=10.0, rho=28.0, beta=8/3):
    x, y, z = xyz
    return [
        sigma * (y - x),
        x * (rho - z) - y,
        x * y - beta * z
    ]

def generate_lorenz_series(x0, steps, dt=0.01):
    t_eval = np.arange(0, steps * dt, dt)
    sol = solve_ivp(lorenz, (0, steps * dt), x0, t_eval=t_eval, method="RK45")
    return sol.y.T.astype(np.float32)  # shape [steps, 3]

def make_dataset(num_series, steps, dt=0.01, context=64):
    all_trajs, inits = [], []
    for _ in range(num_series):
        x0 = np.array([
            np.random.uniform(-20, 20),
            np.random.uniform(-20, 20),
            np.random.uniform(10, 40)
        ])
        traj = generate_lorenz_series(x0, steps, dt)
        all_trajs.append(traj)
        inits.append(x0)
    data = np.stack(all_trajs)  # [num_series, steps, 3]
    inits = np.stack(inits)

    # Chunk into non-overlapping windows of length=context
    trunc_steps = (steps // context) * context
    data_trunc = data[:, :trunc_steps, :]
    n_chunks_per_series = trunc_steps // context
    chunks = data_trunc.reshape(num_series * n_chunks_per_series, context, 3)

    return {
        "dt": dt,
        "params": {"rho": 28.0, "sigma": 10.0, "beta": 8/3},
        "x0": torch.tensor(inits, dtype=torch.float32),
        "data": torch.tensor(data, dtype=torch.float32),
        "chunks": torch.tensor(chunks, dtype=torch.float32),
        "description": f"Lorenz dataset with {num_series} series, {steps} steps, context={context}"
    }

if __name__ == "__main__":
    dt = 0.01
    context = 64

    train = make_dataset(2048, 256, dt, context)
    val   = make_dataset(64, 1024, dt, context)
    test  = make_dataset(256, 1024, dt, context)

    torch.save(train, "lorenz_train.pt")
    torch.save(val, "lorenz_val.pt")
    torch.save(test, "lorenz_test.pt")

    print("Saved: lorenz_train.pt, lorenz_val.pt, lorenz_test.pt")
