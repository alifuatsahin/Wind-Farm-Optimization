import numpy as np
import matplotlib.pyplot as plt

def momentum_conserving_superposition(U_in, U_list, V_list=None, W_list=None, max_iter=10, tol=1e-3, plot=False):
    # Convert to arrays
    U_list = [np.asarray(U) for U in U_list]
    
    # 1. Calculate Individual Deficits (u_i_s)
    # Ensure no negative deficits due to numerical noise if U > U_in slightly
    u_s_list = [np.maximum(U_in - U, 0) for U in U_list]
    print(f"Us list: {np.array(u_s_list).shape}")

    # 2. Calculate Individual Convection Velocities (Uc_i)
    uc_list = []
    for U, u_s in zip(U_list, u_s_list):
        den = np.sum(u_s)
        if den < 1e-8:
            uc_list.append(np.mean(U_in))
        else:
            num = np.sum(U * u_s)
            Uc_val = num / den
            uc_list.append(Uc_val)

    print(f"Uc list: {np.array(uc_list).shape}")
    Uc = np.max(uc_list) if uc_list else np.mean(U_in)
    Uc_history = [Uc]
    Us_history = []

    # ---- ITERATION LOOP (Eq 2.7 & 2.9) ----
    for i in range(max_iter):
        if Uc < 1e-6: Uc = 1e-6

        weights = [uc_i / Uc for uc_i in uc_list]
        print(f"Iteration {i+1}: Weights: {weights}")
        weights_sum = sum(weights)
        weights = [w / weights_sum for w in weights] * len(weights) # normalize weights
        Us_total = sum(w * u_s for w, u_s in zip(weights, u_s_list))
        # Us_total = np.minimum(Us_total, U_in * 0.99)  # prevent over-deficit
        Us_history.append(Us_total)

        Uw_total = U_in - Us_total
        
        num = np.sum(Uw_total * Us_total)
        den = np.sum(Us_total)

        if den < 1e-8:
            break

        Uc_new = num / den
        # relaxation = 0.05
        # Uc_new = (1 - relaxation) * Uc + (relaxation * Uc_new)
        Uc_history.append(Uc_new)

        if np.abs(Uc_new - Uc) / np.abs(Uc_new) < tol:
            print(f"Converged in {i+1} iterations. Uc: {Uc_new:.4f}")
            Uc = Uc_new
            break
        Uc = Uc_new
    else:
        print(f"Max iterations reached. Final Uc: {Uc:.4f}")

    # # Final calculation
    # weights = [uc_i / Uc for uc_i in uc_list]
    # Us_total = sum(w * u_s for w, u_s in zip(weights, u_s_list))
    # U_total = U_in - Us_total

    if plot:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.plot(Uc_history, marker='o')
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Convection Velocity Uc")
        ax1.set_title("Convection Velocity Convergence")
        ax1.grid(True)
        for i, Us in enumerate(Us_history):
            ax2.plot(np.mean(Us, axis=0), label=f"Iter {i+1}")
        ax2.set_xlabel("Grid Point Index")
        ax2.set_ylabel("Mean Total Deficit Us")
        ax2.set_title("Total Deficit Evolution")
        ax2.legend()
        ax2.grid(True)
        plt.show()

    # Transverse
    if V_list is not None:
        V_list = [np.asarray(V) for V in V_list]
        V_total = sum((Uc_i / Uc) * V for Uc_i, V in zip(uc_list, V_list))
    else:
        V_total = None
        
    if W_list is not None:
        W_list = [np.asarray(W) for W in W_list]
        W_total = sum((Uc_i / Uc) * W for Uc_i, W in zip(uc_list, W_list))
    else:
        W_total = None

    return Uw_total, V_total, W_total, Uc

def gaussian_wake(Y, Z, center_y, center_z, u_inf, sigma, max_deficit):
    """
    Generates a simple 2D Gaussian wake field.
    """
    r2 = (Y - center_y)**2 + (Z - center_z)**2
    # Deficit profile
    deficit = max_deficit * np.exp(-r2 / (2 * sigma**2))
    # Wake field
    U_wake = u_inf - deficit
    return U_wake, deficit

def create_rotational_field(Y, Z, center_y, center_z, strength):
    """
    Generates dummy V (y-velocity) and W (z-velocity) for a vortex.
    """
    dy = Y - center_y
    dz = Z - center_z
    r2 = dy**2 + dz**2 + 1e-5 # avoid div by zero
    
    # Simple Rankine-like vortex core
    V = -strength * dz * np.exp(-r2/1000)
    W = strength * dy * np.exp(-r2/1000)
    return V, W

# ==========================================
# 3. EXECUTION SCRIPT
# ==========================================

def run_test():
    # Grid Setup (e.g., a slice of the wind farm)
    L = 200
    N = 100
    y = np.linspace(-L/2, L/2, N)
    z = np.linspace(-L/2, L/2, N)
    Y, Z = np.meshgrid(y, z)

    # Freestream velocity
    U_inf_val = 10.0
    U_in = np.full_like(Y, U_inf_val)

    print("--- Generating Test Wakes ---")
    
    # Wake 1: Centered, deep deficit
    wake1, def1 = gaussian_wake(Y, Z, center_y=0, center_z=0, u_inf=U_inf_val, sigma=20, max_deficit=7.0)
    
    # Wake 2: Slightly offset, overlapping Wake 1
    wake2, def2 = gaussian_wake(Y, Z, center_y=0, center_z=0, u_inf=U_inf_val, sigma=20, max_deficit=4.0)

    # Transverse fields (just to test V/W logic)
    V1, W1 = create_rotational_field(Y, Z, -20, 0, 1.0)
    V2, W2 = create_rotational_field(Y, Z, 20, 0, 1.0)

    # Input lists
    U_list = [wake1, wake2]
    V_list = [V1, V2]
    W_list = [W1, W2]

    # --- Run Momentum Conserving Superposition ---
    print("\n--- Running Momentum Conserving Superposition ---")
    U_mc, V_mc, W_mc, Uc_final = momentum_conserving_superposition(
        U_in, U_list, V_list, W_list, max_iter=50, tol=1e-4, plot=True
    )
    # U_mc, V_mc, W_mc, Uc_final = momentum_conserving_superposition_brent(
    #     U_in, U_list, V_list, W_list, tol=1e-4, plot=True
    # )


    # --- Calculate Standard Linear Superposition (for comparison) ---
    # Linear: U_total = U_in - sum(deficits)
    U_rss = U_in - np.sqrt(def1**2 + def2**2)

    # ==========================================
    # 4. VISUALIZATION
    # ==========================================
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    # Row 1: The Inputs
    im0 = axes[0, 0].pcolormesh(Y, Z, wake1, shading='auto', vmin=4, vmax=10, cmap='viridis')
    axes[0, 0].set_title("Input Wake 1")
    axes[0, 1].pcolormesh(Y, Z, wake2, shading='auto', vmin=4, vmax=10, cmap='viridis')
    axes[0, 1].set_title("Input Wake 2")
    
    # Show overlap area concept
    axes[0, 2].pcolormesh(Y, Z, np.sqrt(def1**2 + def2**2), shading='auto', cmap='Reds')
    axes[0, 2].set_title("Sum of Deficits (RSS)")

    # Row 2: The Results
    im3 = axes[1, 0].pcolormesh(Y, Z, U_rss, shading='auto', vmin=4, vmax=10, cmap='viridis')
    axes[1, 0].set_title("RSS Superposition Result")
    
    im4 = axes[1, 1].pcolormesh(Y, Z, U_mc, shading='auto', vmin=4, vmax=10, cmap='viridis')
    axes[1, 1].set_title(f"Momentum Conserving Result\n(Uc = {Uc_final:.2f})")

    # Difference Plot
    diff = U_mc - U_rss
    limit = np.max(np.abs(diff))
    im5 = axes[1, 2].pcolormesh(Y, Z, diff, shading='auto', cmap='coolwarm', vmin=-limit, vmax=limit)
    axes[1, 2].set_title("Difference (MC - Linear)")
    plt.colorbar(im5, ax=axes[1, 2], label="Delta Velocity (m/s)")

    plt.tight_layout()
    plt.show()

    # --- 1D Cross Section Plot ---
    plt.figure(figsize=(10, 5))
    mid_idx = N // 2
    plt.plot(y, U_in[mid_idx, :], 'k--', label="Freestream")
    plt.plot(y, wake1[mid_idx, :], label="Wake 1 Only", alpha=0.5)
    plt.plot(y, wake2[mid_idx, :], label="Wake 2 Only", alpha=0.5)
    plt.plot(y, U_rss[mid_idx, :], 'r--', linewidth=2, label="RSS Sum")
    plt.plot(y, U_mc[mid_idx, :], 'b-', linewidth=2, label="Momentum Conserving")
    
    plt.title(f"Cross Section at Z=0 (Uc={Uc_final:.3f} m/s)")
    plt.xlabel("Y Position (m)")
    plt.ylabel("Velocity U (m/s)")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    run_test()