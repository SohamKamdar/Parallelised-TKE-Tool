import numpy as np
import os
import glob
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import matplotlib.pyplot as plt

#===========================================================================================================================
# --- USER CONFIGURATION ---

# All settings for the three analyses are in this single block.
# 1. General Parameters
DATA_DIR = r"D:\Soham\XCompactSims\JET\JET20"  # Folder containing the .bin files
NX, NY, NZ = 61, 69, 69                        # Grid size
LX, LY, LZ = 6.0, 10.2, 10.2                   # Domain dimensions
DTYPE = np.float64                             # Data type in the binary file

# 2. Parameters for Single-Timestep Analyses
# Used for 1D Spectrum and Polar Intensities
SINGLE_TIME_STEP = 5

# 3. Parameters for Time History Analysis
# Used for the TKE vs. Time plot
TIME_STEP_START = 1
TIME_STEP_END = 10

# 4. File Naming Parameters
FILE_PREFIX_U = 'ux'
FILE_PREFIX_V = 'uy'
FILE_PREFIX_W = 'uz'
FILE_EXTENSION = '.bin'

# 5. Analysis-Specific Parameters
# For 1D Energy Spectrum
SPECTRA_DIRECTION = 'x'  # Choose 'x', 'y', or 'z'

# 6. For Polar Turbulent Intensities
JET_CENTER_Y_IDX = NY // 2
JET_CENTER_Z_IDX = NZ // 2
NUM_RADIAL_BINS = 20
MAX_RADIUS = min(LY, LZ) / 2.0

# 7. Parallel Processing Parameters
# Set the maximum number of CPU cores to use for the parallel analysis.
# Set to 0 to use all available cores minus one.
MAX_CPU_CORES = 11

# --- END OF USER CONFIGURATION ---

#===========================================================================================================================
# --- SHARED HELPER FUNCTIONS ---

def read_bin_data(filename, nx, ny, nz, dtype=np.float64):
    """Reads a binary file with Fortran ordering and reshapes it."""
    try:
        data_flat = np.fromfile(filename, dtype=dtype)
        if data_flat.size != nx * ny * nz:
            raise ValueError(f"Data size mismatch in {os.path.basename(filename)}")
        return data_flat.reshape((nx, ny, nz), order='F')
    except FileNotFoundError:
        return None # Return None to handle gracefully in parallel workers
    except ValueError as e:
        print(f"Error reshaping data: {e}. Check grid dimensions.")
        return None

#===========================================================================================================================
# --- ANALYSIS 1: 1D ENERGY SPECTRUM (Sequential) ---

def calculate_1d_spectrum(u, v, w, L_dir, N_dir, L_perp1, N_perp1, L_perp2, N_perp2, axis):
    """ Wrapper to compute the 1D energy spectrum using manual normalization and rFFT. """

    # --- Calculate velocity fluctuations ---
    u_fluc = u - np.mean(u, axis=axis, keepdims=True)
    v_fluc = v - np.mean(v, axis=axis, keepdims=True)
    w_fluc = w - np.mean(w, axis=axis, keepdims=True)

    # --- Apply Real Fast Fourier Transform (rFFT) ---
    u_hat = np.fft.rfft(u_fluc, axis=axis)
    v_hat = np.fft.rfft(v_fluc, axis=axis)
    w_hat = np.fft.rfft(w_fluc, axis=axis)

    # --- Calculate modal kinetic energy with manual normalization ---
    E_u = (np.abs(u_hat)**2) / (N_dir**2)
    E_v = (np.abs(v_hat)**2) / (N_dir**2)
    E_w = (np.abs(w_hat)**2) / (N_dir**2)
    modal_energy = 0.5 * (E_u + E_v + E_w)
    
    # --- Integrate over the perpendicular plane and scale by area element ---
    integration_axes = tuple(ax for ax in [0, 1, 2] if ax != axis)
    
    # --- Define effective spacing for the perpendicular plane integration ---
    d_perp1 = L_perp1 / N_perp1
    d_perp2 = L_perp2 / N_perp2
    
    E_k = np.sum(modal_energy, axis=integration_axes) * d_perp1 * d_perp2

    # --- Generate the corresponding wavenumber array ---
    d_dir = L_dir/N_dir
    k = np.fft.rfftfreq(N_dir, d=d_dir) * (2 * np.pi)
    
    return k, E_k

def run_1d_spectrum_analysis():
    """Wrapper to run the 1D energy spectrum analysis for a single timestep."""
    
    # --- Construct file paths ---
    print("--- Starting Analysis 1: 1D Energy Spectrum (Sequential) ---")
    u_filepath = os.path.join(DATA_DIR, f"{FILE_PREFIX_U}-{SINGLE_TIME_STEP}{FILE_EXTENSION}")
    v_filepath = os.path.join(DATA_DIR, f"{FILE_PREFIX_V}-{SINGLE_TIME_STEP}{FILE_EXTENSION}")
    w_filepath = os.path.join(DATA_DIR, f"{FILE_PREFIX_W}-{SINGLE_TIME_STEP}{FILE_EXTENSION}")
    
    # --- Load data ---
    u = read_bin_data(u_filepath, NX, NY, NZ, DTYPE)
    v = read_bin_data(v_filepath, NX, NY, NZ, DTYPE)
    w = read_bin_data(w_filepath, NX, NY, NZ, DTYPE)
    
    if any(data is None for data in [u, v, w]):
        print("Aborting 1D Spectrum analysis due to missing file(s).\n")
        return

    # --- Set parameters based on chosen SPECTRA_DIRECTION ---
    if SPECTRA_DIRECTION.lower() == 'x':
        axis, L_dir, N_dir, L_perp1, N_perp1, L_perp2, N_perp2 = 0, LX, NX, LY, NY, LZ, NZ
    elif SPECTRA_DIRECTION.lower() == 'y':
        axis, L_dir, N_dir, L_perp1, N_perp1, L_perp2, N_perp2 = 1, LY, NY, LX, NX, LZ, NZ
    else: # 'z'
        axis, L_dir, N_dir, L_perp1, N_perp1, L_perp2, N_perp2 = 2, LZ, NZ, LX, NX, LY, NY

    # --- Compute spectrum if data was loaded successfully ---
    k, E_k = calculate_1d_spectrum(u, v, w, L_dir, N_dir, L_perp1, N_perp1, L_perp2, N_perp2, axis)
    
    # --- Save the results to a text file ---
    output_data = np.column_stack((k, E_k))
    
    header = (f"One-Dimensional Energy Spectrum E(k{SPECTRA_DIRECTION})\n"
              f"Generated from timestep {SINGLE_TIME_STEP}\n"
              f"Column 1: Wavenumber (k{SPECTRA_DIRECTION})\n"
              f"Column 2: Energy (E_k{SPECTRA_DIRECTION})\n")
    
    output_filename = f"1D_Energy_Spectrum_t{SINGLE_TIME_STEP}.txt"
    output_filepath = os.path.join(DATA_DIR, output_filename)
    np.savetxt(output_filepath, output_data, header=header, fmt='%.8e', comments='# ')
    
    print(f"1D Spectrum data successfully saved to '{output_filepath}'\n")

    plot_1d_spectrum(output_filepath)

#===========================================================================================================================
# --- ANALYSIS 2: POLAR TURBULENT INTENSITIES (Sequential) ---

def run_polar_intensity_analysis():
    """Wrapper to run the polar turbulent intensity analysis for a single timestep."""
    
     # --- Load Cartesian Velocity Data ---
    print("--- Starting Analysis 2: Polar Turbulent Intensities (Sequential) ---")
    u_filepath = os.path.join(DATA_DIR, f"{FILE_PREFIX_U}-{SINGLE_TIME_STEP}{FILE_EXTENSION}")
    v_filepath = os.path.join(DATA_DIR, f"{FILE_PREFIX_V}-{SINGLE_TIME_STEP}{FILE_EXTENSION}")
    w_filepath = os.path.join(DATA_DIR, f"{FILE_PREFIX_W}-{SINGLE_TIME_STEP}{FILE_EXTENSION}")
    
    u_cart = read_bin_data(u_filepath, NX, NY, NZ, DTYPE)
    v_cart = read_bin_data(v_filepath, NX, NY, NZ, DTYPE)
    w_cart = read_bin_data(w_filepath, NX, NY, NZ, DTYPE)
    
    if any(data is None for data in [u_cart, v_cart, w_cart]):
        print("Aborting Polar Intensity analysis due to missing file(s).\n")
        return

    # --- Calculate Normalization Velocity (Uc) ---
    Uc = np.mean(u_cart, axis=0)[JET_CENTER_Y_IDX, JET_CENTER_Z_IDX]
    print(f"Centerline velocity for normalization Uc(t={SINGLE_TIME_STEP}) = {Uc:.4f}")

    # --- Convert Cartesian Velocity Field to Polar ---
    y_coords = np.linspace(-LY/2, LY/2, NY)
    z_coords = np.linspace(-LZ/2, LZ/2, NZ)
    yy, zz = np.meshgrid(y_coords, z_coords, indexing='ij')
    theta = np.arctan2(zz, yy)

    # --- The axial component is the same, transform v and w to radial and azimuthal components ---
    ux_polar = u_cart
    ur_polar = v_cart * np.cos(theta) + w_cart * np.sin(theta)
    utheta_polar = -v_cart * np.sin(theta) + w_cart * np.cos(theta)

    # --- Calculate Fluctuations of Polar Velocities, subtract the spatial mean (along homogeneous x-axis) from polar fields ---
    ux_fluc = ux_polar - np.mean(ux_polar, axis=0, keepdims=True)
    ur_fluc = ur_polar - np.mean(ur_polar, axis=0, keepdims=True)
    utheta_fluc = utheta_polar - np.mean(utheta_polar, axis=0, keepdims=True)

    # --- Compute Turbulent Intensities using rFFT along the homogeneous 'x' direction (axis=0) ---
    ux_hat = np.fft.rfft(ux_fluc, axis=0)
    ur_hat = np.fft.rfft(ur_fluc, axis=0)
    utheta_hat = np.fft.rfft(utheta_fluc, axis=0)
    
    # --- Manual normalization for rfft: sum of squares of coefficients, doubled for non-zero frequencies, divided by Nx^2 ---
    I_xx = 2 * np.sum(np.abs(ux_hat)**2, axis=0) / (NX**2)
    I_rr = 2 * np.sum(np.abs(ur_hat)**2, axis=0) / (NX**2)
    I_tt = 2 * np.sum(np.abs(utheta_hat)**2, axis=0) / (NX**2)
    
    # --- Perform Azimuthal Averaging ---
    radii_map = np.sqrt(yy**2 + zz**2)
    bin_edges = np.linspace(0, MAX_RADIUS, NUM_RADIAL_BINS + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    I_xx_radial = np.zeros(NUM_RADIAL_BINS)
    I_rr_radial = np.zeros(NUM_RADIAL_BINS)
    I_tt_radial = np.zeros(NUM_RADIAL_BINS)
    
    for i in range(NUM_RADIAL_BINS):
        mask = (radii_map >= bin_edges[i]) & (radii_map < bin_edges[i+1])
        if np.any(mask):
            I_xx_radial[i] = np.mean(I_xx[mask])
            I_rr_radial[i] = np.mean(I_rr[mask])
            I_tt_radial[i] = np.mean(I_tt[mask])
    
    # --- Normalize and Save Final Results ---
    I_xx_norm = I_xx_radial / Uc**2
    I_rr_norm = I_rr_radial / Uc**2
    I_tt_norm = I_tt_radial / Uc**2
    
    output_data = np.column_stack((bin_centers,I_xx_norm,I_rr_norm,I_tt_norm))
    
    header = (f"Cylindrical Turbulent Intensities (Normalized by Uc^2 = {Uc**2:.4f})\n"
              f"Generated from timestep {SINGLE_TIME_STEP}\n"
              f"Column 1: Radius (r)\n"
              f"Column 2: <u_x'u_x'> / Uc^2 (Axial)\n"
              f"Column 3: <u_r'u_r'> / Uc^2 (Radial)\n"
              f"Column 4: <u_theta'u_theta'> / Uc^2 (Azimuthal)")
    
    output_filename = f"Polar_MeanNorm_TKE_t{SINGLE_TIME_STEP}.txt"
    output_filepath = os.path.join(DATA_DIR, output_filename)
    np.savetxt(output_filepath, output_data, header=header, fmt='%.8e', comments='# ')

    print(f"Polar Intensity data successfully saved to '{output_filepath}'\n")

    plot_polar_intensities(output_filepath)

#===========================================================================================================================
# --- ANALYSIS 3: TKE TIME HISTORY (Parallel) ---

def tke_from_fft(u, v, w, Lx, Ly, Lz):
    """ Wrapper to compute the TOTAL TKE in the volume by integrating the 1D spectrum in periodic direction. """
    nx, ny, nz = u.shape
    dx, dy, dz = Lx/nx, Ly/ny, Lz/nz # Using FFT convention spacing Lx/Nx

    # --- Remove mean along x for each (y,z) ---
    u_fluc = u - np.mean(u, axis=0, keepdims=True)
    v_fluc = v - np.mean(v, axis=0, keepdims=True)
    w_fluc = w - np.mean(w, axis=0, keepdims=True)
    
    # --- FFT along periodic x ---
    u_hat = np.fft.fft(u_fluc, axis=0)
    v_hat = np.fft.fft(v_fluc, axis=0)
    w_hat = np.fft.fft(w_fluc, axis=0)
    
    # --- Energy spectrum ---
    E_u = (np.abs(u_hat)**2) / (nx**2)
    E_v = (np.abs(v_hat)**2) / (nx**2)
    E_w = (np.abs(w_hat)**2) / (nx**2)
    
    E_total_modes = 0.5 * (E_u + E_v + E_w)
    
    # --- Integrate over (y,z) plane ---
    E_plane = np.sum(E_total_modes, axis=(1, 2)) * dy * dz
    
    # --- Sum over all wavenumbers ---
    total_tke = np.sum(E_plane)
    return total_tke

def process_single_timestep_for_tke(time_step):
    """ WORKER FUNCTION: Reads data for one timestep, computes TKE, and returns the result. """
    
    # ---- Collect file lists ----
    # print(f"  - Starting TKE calculation for timestep {time_step}...")
    u_filepath = os.path.join(DATA_DIR, f"{FILE_PREFIX_U}-{time_step}{FILE_EXTENSION}")
    v_filepath = os.path.join(DATA_DIR, f"{FILE_PREFIX_V}-{time_step}{FILE_EXTENSION}")
    w_filepath = os.path.join(DATA_DIR, f"{FILE_PREFIX_W}-{time_step}{FILE_EXTENSION}")
    
    u = read_bin_data(u_filepath, NX, NY, NZ, DTYPE)
    v = read_bin_data(v_filepath, NX, NY, NZ, DTYPE)
    w = read_bin_data(w_filepath, NX, NY, NZ, DTYPE)
    
    if any(data is None for data in [u, v, w]):
        print(f"    -> Skipping timestep {time_step}, file(s) not found.")
        return None
        
    total_tke = tke_from_fft(u, v, w, LX, LY, LZ)
    print(f"  - TKE for timestep {time_step} = {total_tke:.4e}")
    return (time_step*10, total_tke)

def run_tke_time_history_analysis_parallel():
    """
    MANAGER FUNCTION: Sets up a process pool and distributes the TKE calculation
    for each timestep to the worker processes.
    """
    print("--- Starting Analysis 3: TKE Time History (Parallel) ---")
    
    timesteps_to_process = range(TIME_STEP_START, TIME_STEP_END + 1)
    time_history = []
    
    # --- Determine number of CPU cores to use ---
    available_cores = multiprocessing.cpu_count()
    if MAX_CPU_CORES > 0:
        num_cores = min(MAX_CPU_CORES, available_cores)
    else:
        num_cores = max(1, available_cores - 1)

    print(f"Using {num_cores} of {available_cores} CPU cores for parallel processing.")

    # --- Set up the process pool and distribute work ---
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        future_to_timestep = {executor.submit(process_single_timestep_for_tke, ts): ts for ts in timesteps_to_process}
        
        for future in as_completed(future_to_timestep):
            result = future.result()
            if result is not None:
                time_history.append(result)

    if not time_history:
        print("Aborting TKE Time History analysis, no data was successfully processed.\n")
        return
        
    time_history.sort()
    
    output_data = np.array(time_history)
    header = (f"Time History of Total Turbulent Kinetic Energy (TKE)\n"
              f"Generated from timesteps {TIME_STEP_START} to {TIME_STEP_END}\n"
              f"Column 1: Timestep\n"
              f"Column 2: Total TKE")
    
    output_filename = f"Time_History_TKE_t{TIME_STEP_START}-{TIME_STEP_END}.txt"
    output_filepath = os.path.join(DATA_DIR, output_filename)
    np.savetxt(output_filepath, output_data, header=header, fmt='%d %.8e', comments='# ')
    
    print(f"TKE Time History data successfully saved to '{output_filepath}'\n")

    plot_tke_history(output_filepath)

#===========================================================================================================================
# --- PLOTTING FUNCTIONS ---

def plot_tke_history(data_filepath):
    """Plots TKE vs. Time History."""
    data = np.loadtxt(data_filepath, comments='#')
    time_steps = data[:, 0]
    tke_values = data[:, 1]

    # --- Create the plot ---
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(time_steps, tke_values, 'k-')                       # Black solid line (k-), to set markers use marker = 'o', markersize = 4
    ax.set_title('Turbulent Kinetic Energy vs. Time')
    ax.set_xlabel('Non-Dimensional Time (t)')
    ax.set_ylabel('Mean Turbulent Kinetic Energy')
    ax.set_xlim(0, 100)                                         # Set x-limits from 0 to 100
    ax.set_ylim(0, 0.25)                                        # Set y-limits from 0 to 0.25
    ax.set_xticks(np.arange(0, 101, 10))                        # Set x-ticks at intervals of 10 between 0 and 100
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)  # Grid with dashed lines
   
    plot_filename = os.path.splitext(data_filepath)[0] + '.png'
    plt.savefig(plot_filename, dpi=300)
    plt.close()
    
    print(f"Plot saved to '{plot_filename}'\n")

def plot_1d_spectrum(data_filepath):
    """Plots 1D Energy Spectrum."""
    data = np.loadtxt(data_filepath, comments='#')
    k = data[:, 0]
    E_k = data[:, 1]
    valid_indices = k > 0
    k = k[valid_indices]
    E_k = E_k[valid_indices]
    
    # --- Create the plot ---
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.loglog(k, E_k, 'k-')                                     # Black solid line (k-)
    ax.set_title('1D Energy Spectrum')
    ax.set_xlabel(f'Wavenumber (k{SPECTRA_DIRECTION})')
    ax.set_ylabel(f'Energy Spectrum (E(k{SPECTRA_DIRECTION}))')
    ax.set_xlim(1, 30)                                          # Set x-limits from 1 to 30
    ax.set_ylim(1e-6, 1e-1)                                     # Set y-limits from 1e-6 to 1e-1       
    ax.set_xticks(np.arange(5, 31, 5))                          # Set x-ticks at intervals of 5 between 5 and 30
    ax.set_xticklabels([str(i) for i in range(5, 31, 5)])       # Normal Number ticks
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)  # Grid with dashed lines
    
    plot_filename = os.path.splitext(data_filepath)[0] + '.png'
    plt.savefig(plot_filename, dpi=300)
    plt.close()
    
    print(f"Plot saved to '{plot_filename}'\n")

def plot_polar_intensities(data_filepath):
    """Plots the three polar turbulent intensity profiles."""
    data = np.loadtxt(data_filepath, comments='#')
    radius = data[:, 0]
    I_xx = data[:, 1]
    I_rr = data[:, 2]
    I_tt = data[:, 3]
    radius_norm = radius / 1.0
    
    # --- Create the plot ---
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(radius_norm, I_xx, 'k-', label="Axial <ux'ux'> / Uc²")        # Black solid line (k-)
    ax.plot(radius_norm, I_rr, 'k--', label="Radial <ur'ur'> / Uc²")      # Black dashed line (k--)
    ax.plot(radius_norm, I_tt, 'k-.', label="Azimuthal <uθ'uθ'> / Uc²")   # Black dash-dot line (k-.)
    ax.set_title('Mean Normalized Turbulent Intensities')
    ax.set_xlabel('Normalized Radius (r/R)')
    ax.set_ylabel('Turbulent Intensity')
    ax.set_xlim(0, 5)                                                       # Set x-limits from 0 to 5
    ax.set_ylim(0, 0.14)                                                    # Set y-limits from 0 to 0.14
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)              # Grid with dashed lines
    ax.legend()                                                             # Show legend
    
    plot_filename = os.path.splitext(data_filepath)[0] + '.png'
    plt.savefig(plot_filename, dpi=300)
    plt.close()
    
    print(f"Plot saved to '{plot_filename}'\n")


#===========================================================================================================================
# --- MAIN EXECUTION ---
if __name__ == '__main__':
    run_1d_spectrum_analysis()
    run_polar_intensity_analysis()
    run_tke_time_history_analysis_parallel()
    print("--- All analyses complete. --- \n")
