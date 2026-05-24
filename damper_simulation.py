"""
Fluid Damper Theoretical Models -- Simulation & Performance Analysis
lianghuaze
Models:
  1. Linear viscous
  2. Nonlinear viscous (power-law)
  3. Kelvin-Voigt (spring + dashpot in parallel)
  4. Maxwell (spring + dashpot in series, ODE)
  5. Bouc-Wen (hysteretic, ODE)

All models are driven by the same sinusoidal displacement excitation.
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
import sys

# Ensure UTF-8 output on Windows terminals
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# -- Global settings ----------------------------------------------------------
plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "savefig.bbox": "tight",
    "savefig.dpi": 200,
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
    "axes.unicode_minus": False,
})

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# -- Excitation ---------------------------------------------------------------
def generate_excitation(X0, f, dt, n_cycles, settle_cycles=1):
    """Sinusoidal displacement and velocity time histories."""
    T = 1.0 / f
    t_total = n_cycles * T
    t = np.arange(0, t_total, dt)
    omega = 2 * np.pi * f
    x = X0 * np.sin(omega * t)
    v = omega * X0 * np.cos(omega * t)
    N_settle = int(settle_cycles * T / dt)
    return t, x, v, N_settle


# =============================================================================
#  Model 1 - Linear Viscous
# =============================================================================
def linear_viscous(v, c):
    return c * v


# =============================================================================
#  Model 2 - Nonlinear Viscous (Power-Law)
# =============================================================================
def nonlinear_viscous(v, c, alpha):
    return c * np.sign(v) * np.abs(v) ** alpha


# =============================================================================
#  Model 3 - Kelvin-Voigt
# =============================================================================
def kelvin_voigt(x, v, k, c):
    return k * x + c * v


# =============================================================================
#  Model 4 - Maxwell  (ODE: dF/dt = (c*v - F) / lambda,  lambda = c/k)
# =============================================================================
def maxwell_ode(t, F, v_func, c, lam):
    v = v_func(t)
    return (c * v - F) / lam


def simulate_maxwell(t, v, c, k):
    lam = c / k
    v_interp = lambda tau: np.interp(tau, t, v)
    sol = solve_ivp(
        maxwell_ode,
        [t[0], t[-1]],
        [0.0],
        t_eval=t,
        args=(v_interp, c, lam),
        method="RK45",
        rtol=1e-9,
        atol=1e-12,
    )
    return sol.y[0]


# =============================================================================
#  Model 5 - Bouc-Wen
#    F  = c*v + k*x + alpha_z*z
#    dz = A*v - beta*|v|*|z|^(n-1)*z - gamma*v*|z|^n
# =============================================================================
def bouc_wen_ode(t, z, v_func, A, beta, gamma, n):
    v = v_func(t)
    dz = A * v - beta * np.abs(v) * np.abs(z) ** (n - 1) * z - gamma * v * np.abs(z) ** n
    return dz


def simulate_bouc_wen(t, x, v, c, k, alpha_z, A, beta, gamma, n):
    v_interp = lambda tau: np.interp(tau, t, v)
    sol = solve_ivp(
        bouc_wen_ode,
        [t[0], t[-1]],
        [0.0],
        t_eval=t,
        args=(v_interp, A, beta, gamma, n),
        method="RK45",
        rtol=1e-9,
        atol=1e-12,
    )
    z = sol.y[0]
    F = c * v + k * x + alpha_z * z
    return F, z


# -- Performance metrics ------------------------------------------------------
def compute_metrics(t, x, v, F, N_settle):
    """Extract the last complete cycle and compute damper metrics."""
    idx_settle = max(0, N_settle)
    x_trim = x[idx_settle:]
    zero_crossings = np.where(np.diff(np.sign(x_trim)) > 0)[0]
    if len(zero_crossings) < 2:
        return None

    start = idx_settle + zero_crossings[-2]
    end = idx_settle + zero_crossings[-1]

    t_cyc = t[start:end]
    x_cyc = x[start:end]
    v_cyc = v[start:end]
    F_cyc = F[start:end]

    # Energy dissipated per cycle: Ed = contour_integral F dx = integral F*v dt
    Ed = np.trapezoid(F_cyc * v_cyc, t_cyc)

    # Maximum force
    F_max = np.max(np.abs(F_cyc))

    # Equivalent stiffness from peak-to-peak
    i_max, i_min = np.argmax(x_cyc), np.argmin(x_cyc)
    K_eq = (F_cyc[i_max] - F_cyc[i_min]) / (x_cyc[i_max] - x_cyc[i_min])

    X0 = x_cyc[i_max]

    # Storage stiffness (secant stiffness at max displacement)
    K_storage = np.abs(F_cyc[i_max] / X0) if np.abs(X0) > 1e-12 else 0.0

    # Equivalent viscous damping ratio: xi_eq = Ed / (4*pi*E_stored)
    # where E_stored = 0.5 * K_storage * X0^2
    # For models with negligible storage stiffness, xi_eq -> inf
    min_K = 1e2  # threshold below which stiffness is considered negligible
    if K_storage > min_K:
        xi_eq = Ed / (2 * np.pi * K_storage * X0 ** 2)
    else:
        xi_eq = np.inf

    # Loss factor: eta = Ed / (pi * K_eq * X0^2)
    if K_eq > min_K:
        eta = Ed / (np.pi * K_eq * X0 ** 2)
    else:
        eta = np.inf

    return {
        "Ed": Ed,
        "F_max": F_max,
        "K_eq": K_eq,
        "K_storage": K_storage,
        "xi_eq": xi_eq,
        "eta": eta,
        "x_max": X0,
        "x_min": x_cyc[i_min],
        "cycle_data": (t_cyc, x_cyc, v_cyc, F_cyc),
    }


# -- Chirp / Frequency Sweep Excitation ---------------------------------------
def generate_chirp(X0, f_start, f_end, duration, dt):
    """Linear frequency sweep (chirp) signal.

    Instantaneous frequency ramps linearly from f_start to f_end.
    Returns t, x, v, f_inst arrays.
    """
    t = np.arange(0, duration, dt)
    f_inst = f_start + (f_end - f_start) * t / duration
    phase = 2 * np.pi * (f_start * t + 0.5 * (f_end - f_start) * t ** 2 / duration)
    x = X0 * np.sin(phase)
    v = 2 * np.pi * X0 * f_inst * np.cos(phase)
    return t, x, v, f_inst


# -- Plotting -----------------------------------------------------------------
def plot_all(models_results, params):
    """Generate four-panel comparison figure."""
    fig = plt.figure(figsize=(14, 11))
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.30)

    colors = {
        "线性粘性 (Linear Viscous)": "#1f77b4",
        "非线性粘性 (Nonlinear)": "#ff7f0e",
        "Kelvin-Voigt": "#2ca02c",
        "Maxwell": "#d62728",
        "Bouc-Wen": "#9467bd",
    }

    ax_fd = fig.add_subplot(gs[0, 0])
    ax_fv = fig.add_subplot(gs[0, 1])
    ax_ft = fig.add_subplot(gs[1, 0])
    ax_xt = fig.add_subplot(gs[1, 1])

    for name, (name_short, F) in models_results.items():
        color = colors.get(name, "gray")
        T = 1.0 / params["f"]
        n_plot = int(2 * T / params["dt"])
        t_plot = params["t"][-n_plot:]
        x_plot = params["x"][-n_plot:]
        v_plot = params["v"][-n_plot:]
        F_plot = F[-n_plot:]

        ax_fd.plot(x_plot * 1e3, F_plot * 1e-3, color=color, lw=1.2, label=name)
        ax_fv.plot(v_plot, F_plot * 1e-3, color=color, lw=1.2, label=name)
        ax_ft.plot(t_plot, F_plot * 1e-3, color=color, lw=1.2, label=name)

    T = 1.0 / params["f"]
    n_plot = int(2 * T / params["dt"])
    t_plot = params["t"][-n_plot:]
    x_plot = params["x"][-n_plot:]
    ax_xt.plot(t_plot, x_plot * 1e3, color="black", lw=1.5, label="位移")

    ax_fd.set_xlabel("位移 [mm]")
    ax_fd.set_ylabel("阻尼力 [kN]")
    ax_fd.set_title("力-位移 滞回曲线")
    ax_fd.legend(fontsize=7, loc="upper left")
    ax_fd.grid(True, alpha=0.3)

    ax_fv.set_xlabel("速度 [m/s]")
    ax_fv.set_ylabel("阻尼力 [kN]")
    ax_fv.set_title("力-速度 曲线")
    ax_fv.legend(fontsize=7)
    ax_fv.grid(True, alpha=0.3)

    ax_ft.set_xlabel("时间 [s]")
    ax_ft.set_ylabel("阻尼力 [kN]")
    ax_ft.set_title("力 时程曲线")
    ax_ft.legend(fontsize=7)
    ax_ft.grid(True, alpha=0.3)

    ax_xt.set_xlabel("时间 [s]")
    ax_xt.set_ylabel("位移 [mm]")
    ax_xt.set_title("激励位移")
    ax_xt.legend(fontsize=7)
    ax_xt.grid(True, alpha=0.3)

    fig.suptitle(
        f"流体阻尼器模型对比  |  "
        f"X0={params['X0']*1e3:.0f} mm,  f={params['f']:.1f} Hz",
        fontsize=13,
        y=1.01,
    )
    path = os.path.join(OUTPUT_DIR, "model_comparison.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  已保存: {path}")


def plot_individual_hysteresis(models_results, metrics, params):
    """One hysteresis plot per model with metrics annotation."""
    n = len(models_results)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows))
    if n == 1:
        axes = np.array([[axes]])
    elif axes.ndim == 1:
        axes = axes.reshape(1, -1)
    colors = {
        "线性粘性 (Linear Viscous)": "#1f77b4",
        "非线性粘性 (Nonlinear)": "#ff7f0e",
        "Kelvin-Voigt": "#2ca02c",
        "Maxwell": "#d62728",
        "Bouc-Wen": "#9467bd",
    }

    for idx, (name, (_, F)) in enumerate(models_results.items()):
        ax = axes[idx // ncols, idx % ncols]
        color = colors.get(name, "gray")
        met = metrics[name]
        if met is None:
            ax.set_title(f"{name}\n(无循环数据)")
            continue
        t_cyc, x_cyc, v_cyc, F_cyc = met["cycle_data"]

        ax.plot(x_cyc * 1e3, F_cyc * 1e-3, color=color, lw=1.5)
        ax.fill(x_cyc * 1e3, F_cyc * 1e-3, color=color, alpha=0.15)
        ax.set_xlabel("位移 [mm]")
        ax.set_ylabel("阻尼力 [kN]")
        ax.set_title(name)
        ax.grid(True, alpha=0.3)

        xi_str = "inf" if np.isinf(met['xi_eq']) else f"{met['xi_eq']*100:.1f}"
        eta_str = "inf" if np.isinf(met['eta']) else f"{met['eta']:.3f}"
        textstr = (
            f"Ed = {met['Ed']:.2f} J\n"
            f"Fmax = {met['F_max']*1e-3:.2f} kN\n"
            f"Keq = {met['K_eq']*1e-6:.3f} MN/m\n"
            f"xi_eq = {xi_str} %\n"
            f"eta = {eta_str}"
        )
        ax.text(
            0.05, 0.95, textstr, transform=ax.transAxes, fontsize=8,
            verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    # Hide unused subplots
    for idx in range(n, nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)

    fig.suptitle(
        f"各模型滞回曲线  |  "
        f"X0={params['X0']*1e3:.0f} mm,  f={params['f']:.1f} Hz",
        fontsize=13,
    )
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "individual_hysteresis.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  已保存: {path}")


# -- Simulation helper --------------------------------------------------------
def simulate_all_models(t, x, v, c, k, alpha, alpha_z_bw, A_bw, beta_bw, gamma_bw, n_bw):
    """Run all five damper models on the given excitation. Returns {name: (tag, F)}."""
    results = {}
    results["线性粘性 (Linear Viscous)"] = ("LV", linear_viscous(v, c))
    results["非线性粘性 (Nonlinear)"] = ("NLV", nonlinear_viscous(v, c, alpha))
    results["Kelvin-Voigt"] = ("KV", kelvin_voigt(x, v, k, c))
    results["Maxwell"] = ("MX", simulate_maxwell(t, v, c, k))
    F_bw, _ = simulate_bouc_wen(t, x, v, c, k, alpha_z_bw, A_bw, beta_bw, gamma_bw, n_bw)
    results["Bouc-Wen"] = ("BW", F_bw)
    return results


# -- Scenario 2: Chirp / Frequency Sweep --------------------------------------
def run_chirp_sweep(c, k, alpha, A_bw, beta_bw, gamma_bw, n_bw):
    print("\n" + "=" * 65)
    print("  工况2: 扫频激励 (Chirp Signal)")
    print("=" * 65)

    X0 = 0.025
    f_start, f_end = 0.1, 5.0
    duration = 20.0
    dt = 1e-3

    print(f"  X0={X0*1e3:.0f} mm,  频率 {f_start} -> {f_end} Hz,  持续 {duration} s,  dt={dt*1e3:.0f} ms")

    t, x, v, f_inst = generate_chirp(X0, f_start, f_end, duration, dt)
    alpha_z_bw = k * X0 * 0.6

    print("  正在仿真所有模型 ...", end=" ")
    results = simulate_all_models(t, x, v, c, k, alpha, alpha_z_bw, A_bw, beta_bw, gamma_bw, n_bw)
    print("完成")

    # -- Plot: 4-panel comparison with color-mapped frequency -----------------
    fig = plt.figure(figsize=(15, 12))
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.30)

    colors = {
        "线性粘性 (Linear Viscous)": "#1f77b4",
        "非线性粘性 (Nonlinear)": "#ff7f0e",
        "Kelvin-Voigt": "#2ca02c",
        "Maxwell": "#d62728",
        "Bouc-Wen": "#9467bd",
    }

    ax_fd = fig.add_subplot(gs[0, 0])
    ax_fv = fig.add_subplot(gs[0, 1])
    ax_ft = fig.add_subplot(gs[1, 0])
    ax_if = fig.add_subplot(gs[1, 1])

    # Downsample for cleaner scatter plots
    step = max(1, len(t) // 4000)

    for name, (_, F) in results.items():
        color = colors.get(name, "gray")
        f_norm = (f_inst - f_start) / (f_end - f_start)

        ax_fd.scatter(x[::step] * 1e3, F[::step] * 1e-3, c=f_norm[::step],
                      cmap="plasma", s=2, alpha=0.6, label=name)
        ax_fv.scatter(v[::step], F[::step] * 1e-3, c=f_norm[::step],
                      cmap="plasma", s=2, alpha=0.6, label=name)
        ax_ft.plot(t, F * 1e-3, color=color, lw=0.8, label=name)

    ax_fd.set_xlabel("位移 [mm]")
    ax_fd.set_ylabel("阻尼力 [kN]")
    ax_fd.set_title("力-位移 扫频响应 (颜色=瞬时频率)")
    ax_fd.legend(fontsize=7, loc="upper left")
    ax_fd.grid(True, alpha=0.3)

    ax_fv.set_xlabel("速度 [m/s]")
    ax_fv.set_ylabel("阻尼力 [kN]")
    ax_fv.set_title("力-速度 扫频响应 (颜色=瞬时频率)")
    ax_fv.legend(fontsize=7)
    ax_fv.grid(True, alpha=0.3)

    ax_ft.set_xlabel("时间 [s]")
    ax_ft.set_ylabel("阻尼力 [kN]")
    ax_ft.set_title("力 时程曲线")
    ax_ft.legend(fontsize=7)
    ax_ft.grid(True, alpha=0.3)

    ax_if.plot(t, f_inst, color="black", lw=1.5)
    ax_if.set_xlabel("时间 [s]")
    ax_if.set_ylabel("瞬时频率 [Hz]")
    ax_if.set_title("瞬时频率")
    ax_if.grid(True, alpha=0.3)

    fig.suptitle(
        f"扫频激励对比  |  X0={X0*1e3:.0f} mm,  "
        f"f: {f_start} -> {f_end} Hz,  T={duration:.0f} s",
        fontsize=13, y=1.01,
    )
    path = os.path.join(OUTPUT_DIR, "chirp_comparison.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  已保存: {path}")

    # -- Plot: individual model sweep hysteresis -------------------------------
    n = len(results)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows))
    if n == 1:
        axes = np.array([[axes]])
    elif axes.ndim == 1:
        axes = axes.reshape(1, -1)

    for idx, (name, (_, F)) in enumerate(results.items()):
        ax = axes[idx // ncols, idx % ncols]
        color = colors.get(name, "gray")
        f_norm = (f_inst - f_start) / (f_end - f_start)

        sc = ax.scatter(x[::step] * 1e3, F[::step] * 1e-3, c=f_norm[::step],
                        cmap="plasma", s=2, alpha=0.6)
        ax.set_xlabel("位移 [mm]")
        ax.set_ylabel("阻尼力 [kN]")
        ax.set_title(name)
        ax.grid(True, alpha=0.3)
        cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("归一化频率", fontsize=7)

    for idx in range(n, nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)

    fig.suptitle(
        f"各模型扫频滞回曲线  |  X0={X0*1e3:.0f} mm,  "
        f"f: {f_start} -> {f_end} Hz",
        fontsize=13,
    )
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "chirp_individual_hysteresis.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  已保存: {path}")


# -- Scenario 3: Multi-Frequency Comparison -----------------------------------
def run_multi_freq(c, k, alpha, A_bw, beta_bw, gamma_bw, n_bw):
    print("\n" + "=" * 65)
    print("  工况3: 多频率对比")
    print("=" * 65)

    X0 = 0.025
    freqs = [0.2, 0.5, 1.0, 2.0, 3.0, 5.0]
    dt = 2e-4
    n_cycles = 8
    settle_cycles = 2

    model_names = [
        "线性粘性 (Linear Viscous)",
        "非线性粘性 (Nonlinear)",
        "Kelvin-Voigt",
        "Maxwell",
        "Bouc-Wen",
    ]
    model_colors = {
        "线性粘性 (Linear Viscous)": "#1f77b4",
        "非线性粘性 (Nonlinear)": "#ff7f0e",
        "Kelvin-Voigt": "#2ca02c",
        "Maxwell": "#d62728",
        "Bouc-Wen": "#9467bd",
    }
    model_markers = {
        "线性粘性 (Linear Viscous)": "o",
        "非线性粘性 (Nonlinear)": "s",
        "Kelvin-Voigt": "D",
        "Maxwell": "^",
        "Bouc-Wen": "v",
    }

    # Collect metrics for each model at each frequency
    all_metrics = {name: {"f": [], "Ed": [], "F_max": [], "K_eq": [], "xi_eq": [], "eta": []}
                   for name in model_names}

    for fi, freq in enumerate(freqs):
        print(f"  频率 {freq:.1f} Hz ({fi+1}/{len(freqs)}) ...", end=" ")
        t, x, v, N_settle = generate_excitation(X0, freq, dt, n_cycles, settle_cycles)
        alpha_z_bw = k * X0 * 0.6
        results = simulate_all_models(t, x, v, c, k, alpha, alpha_z_bw, A_bw, beta_bw, gamma_bw, n_bw)

        for name, (_, F) in results.items():
            met = compute_metrics(t, x, v, F, N_settle)
            if met is not None:
                all_metrics[name]["f"].append(freq)
                all_metrics[name]["Ed"].append(met["Ed"])
                all_metrics[name]["F_max"].append(met["F_max"] * 1e-3)
                all_metrics[name]["K_eq"].append(met["K_eq"] * 1e-6)
                all_metrics[name]["xi_eq"].append(
                    met["xi_eq"] if not np.isinf(met["xi_eq"]) else np.nan
                )
                all_metrics[name]["eta"].append(
                    met["eta"] if not np.isinf(met["eta"]) else np.nan
                )
        print("完成")

    # -- Print summary table ---------------------------------------------------
    print("\n" + "-" * 100)
    print(f"  {'频率':>6s} | {'模型':<28s} | {'Ed [J]':>10s} | {'Fmax [kN]':>10s} | "
          f"{'Keq [MN/m]':>12s} | {'xi_eq [%]':>10s} | {'eta':>8s}")
    print("-" * 100)
    for freq in freqs:
        first = True
        for name in model_names:
            idx = list(all_metrics[name]["f"]).index(freq)
            xi_val = all_metrics[name]["xi_eq"][idx]
            eta_val = all_metrics[name]["eta"][idx]
            xi_str = "    inf" if np.isnan(xi_val) else f"{xi_val*100:7.1f}"
            eta_str = "    inf" if np.isnan(eta_val) else f"{eta_val:7.3f}"
            label = f"{freq:5.1f} Hz" if first else "      "
            print(f"  {label:>6s} | {name:<28s} | {all_metrics[name]['Ed'][idx]:10.2f} | "
                  f"{all_metrics[name]['F_max'][idx]:10.2f} | "
                  f"{all_metrics[name]['K_eq'][idx]:12.3f} | {xi_str:>10s} | {eta_str:>8s}")
            first = False
        print("-" * 100)

    # -- Plot: 4-panel frequency-dependent metrics -----------------------------
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    metric_keys = [
        ("Ed", "每循环耗能 Ed [J]"),
        ("K_eq", "等效刚度 Keq [MN/m]"),
        ("xi_eq", "等效阻尼比 xi_eq [%]"),
        ("F_max", "最大阻尼力 Fmax [kN]"),
    ]

    for (key, ylabel), ax in zip(metric_keys, axes.flat):
        for name in model_names:
            f_vals = np.array(all_metrics[name]["f"])
            m_vals = np.array(all_metrics[name][key])
            if key == "xi_eq":
                m_vals = m_vals * 100  # convert to percent
            valid = ~np.isnan(m_vals)
            if np.any(valid):
                ax.plot(f_vals[valid], m_vals[valid],
                        color=model_colors[name], marker=model_markers[name],
                        markersize=6, lw=1.5, label=name, markerfacecolor="white")
        ax.set_xlabel("频率 [Hz]")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=6)

    fig.suptitle(
        f"阻尼器性能指标随频率变化  |  X0={X0*1e3:.0f} mm",
        fontsize=13,
    )
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "multi_freq_metrics.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"\n  已保存: {path}")

    # -- Plot: hysteresis overlay at each frequency for Maxwell & Bouc-Wen ----
    fig, axes = plt.subplots(2, len(freqs), figsize=(3.2 * len(freqs), 7))
    for row, (name, tag) in enumerate([("Maxwell", "MX"), ("Bouc-Wen", "BW")]):
        for col, freq in enumerate(freqs):
            ax = axes[row, col]
            t, x, v, N_settle = generate_excitation(X0, freq, dt, n_cycles, settle_cycles)
            alpha_z_bw = k * X0 * 0.6
            res = simulate_all_models(t, x, v, c, k, alpha, alpha_z_bw, A_bw, beta_bw, gamma_bw, n_bw)
            _, F = res[name]

            # Last 2 cycles
            T_cyc = 1.0 / freq
            n_last = int(2 * T_cyc / dt)
            ax.plot(x[-n_last:] * 1e3, F[-n_last:] * 1e-3,
                    color=model_colors[name], lw=1.0)
            ax.fill(x[-n_last:] * 1e3, F[-n_last:] * 1e-3,
                    color=model_colors[name], alpha=0.1)
            if row == 0:
                ax.set_title(f"{freq:.1f} Hz", fontsize=9)
            if col == 0:
                ax.set_ylabel(f"{name}\n阻尼力 [kN]", fontsize=8)
            if row == 1:
                ax.set_xlabel("位移 [mm]", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.3)

    fig.suptitle("Maxwell 与 Bouc-Wen 滞回曲线随频率演化", fontsize=13)
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "multi_freq_hysteresis.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  已保存: {path}")


# -- Main ---------------------------------------------------------------------
def main():
    # -- Damper parameters (shared across all scenarios) ------------------------
    c = 80e3       # viscous coefficient [N*s/m]
    k = 2000e3     # stiffness [N/m]
    alpha = 0.4    # nonlinear exponent

    # Bouc-Wen parameters
    A_bw = 1.0
    beta_bw = 0.5
    gamma_bw = 0.5
    n_bw = 2

    # =========================================================================
    #  工况1: 单频正弦激励
    # =========================================================================
    print("=" * 65)
    print("  工况1: 单频正弦激励")
    print("=" * 65)

    X0 = 0.025
    f = 1.0
    dt = 5e-4
    n_cycles = 6
    settle_cycles = 1

    print(f"  位移幅值 X0 = {X0*1e3:.0f} mm,  频率 f = {f:.1f} Hz,  循环数 = {n_cycles}")

    t, x, v, N_settle = generate_excitation(X0, f, dt, n_cycles, settle_cycles)
    params = {"X0": X0, "f": f, "dt": dt, "n_cycles": n_cycles,
              "t": t, "x": x, "v": v}
    alpha_z_bw = k * X0 * 0.6

    print("  正在仿真所有模型 ...", end=" ")
    results = simulate_all_models(t, x, v, c, k, alpha, alpha_z_bw, A_bw, beta_bw, gamma_bw, n_bw)
    print("完成")

    # Metrics
    print("\n" + "-" * 65)
    print("  性能指标 (基于最后一个稳态循环)")
    print("-" * 65)

    metrics = {}
    for name, (_, F) in results.items():
        met = compute_metrics(t, x, v, F, N_settle)
        metrics[name] = met

    header = (f"{'模型':<30s} {'Ed [J]':>9s}  {'Fmax [kN]':>10s}  "
              f"{'Keq [MN/m]':>12s}  {'xi_eq [%]':>9s}  {'eta':>8s}")
    print(header)
    print("-" * len(header))

    for name, met in metrics.items():
        if met is None:
            print(f"{name:<30s}  {'N/A':>9s}")
            continue
        xi_str = "inf" if np.isinf(met['xi_eq']) else f"{met['xi_eq']*100:9.1f}"
        eta_str = "inf" if np.isinf(met['eta']) else f"{met['eta']:8.3f}"
        print(
            f"{name:<30s}  {met['Ed']:9.2f}  {met['F_max']*1e-3:10.2f}  "
            f"{met['K_eq']*1e-6:12.3f}  {xi_str:>9s}  {eta_str:>8s}"
        )

    print("-" * 65)
    omega = 2 * np.pi * f
    E_input = np.pi * c * omega * X0 ** 2
    print(f"  参考值: 线性粘性 Ed = pi*c*omega*X0^2 = {E_input:.2f} J")
    print("-" * 65)

    print("\n正在生成图表 ...")
    plot_all(results, params)
    plot_individual_hysteresis(results, metrics, params)
    print("完成。")

    # =========================================================================
    #  工况2: 扫频激励
    # =========================================================================
    run_chirp_sweep(c, k, alpha, A_bw, beta_bw, gamma_bw, n_bw)

    # =========================================================================
    #  工况3: 多频率对比
    # =========================================================================
    run_multi_freq(c, k, alpha, A_bw, beta_bw, gamma_bw, n_bw)

    print("\n" + "=" * 65)
    print("  所有工况完成。")
    print("=" * 65)


if __name__ == "__main__":
    main()
