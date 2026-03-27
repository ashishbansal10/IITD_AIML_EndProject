import subprocess
import sys
import os
import shutil
import platform
import gc
import warnings
from importlib import metadata

# Try to import packaging; install it if missing
try:
    from packaging.specifiers import SpecifierSet
    from packaging.version import Version
except ImportError:
    print("🛠️ Installing 'packaging' helper...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "packaging", "--quiet"])
    from packaging.specifiers import SpecifierSet
    from packaging.version import Version


# Silence PyTorch XPU warnings specifically at the top level
# This catches the warning during the initial 'import torch'
warnings.filterwarnings("ignore", category=UserWarning, module="torch.xpu")
warnings.filterwarnings("ignore", message=".*not officially supported.*")


# Internal constants
_LOCAL_REQ_FILE = "requirements.txt"
_GLOBAL_REQ_FILE = r"C:\Users\ashish\Documents\Ashish\anaconda\ai_lab_312\requirements.txt"

# Public API - Cleanly separated
__all__ = [
    # Core Environment Management
    'setup_requirements', 'check_requirements', 'manage_env', 'prepare_for_deployment',
    # Colab Drive Sync
    'sync_to_colab_drive',
    # Hardware Utilities
    'get_cpu_info',
    # TensorFlow Utilities
    'apply_tf_threading', 'setup_tensorflow',
    # PyTorch Utilities
    'apply_torch_threading', 'torch_hard_reset', 'get_pytorch_device', 'setup_pytorch'
]

# --- CORE ENVIRONMENT MANAGER ---

def _get_req_path(user_path=None):
    if user_path and os.path.exists(user_path): return user_path
    if os.path.exists(_LOCAL_REQ_FILE): return os.path.abspath(_LOCAL_REQ_FILE)
    if os.path.exists(_GLOBAL_REQ_FILE):
        print(f"ℹ️ Local requirements not found. Falling back to global: {_GLOBAL_REQ_FILE}")
        return _GLOBAL_REQ_FILE
    return None

def _parse_line(line):
    for op in ['==', '>=', '<=', '~=']:
        if op in line:
            name, v_spec = line.split(op, 1)
            name = name.strip().split('[')[0].replace('_', '-').lower()
            return name, op + v_spec.strip()
    return line.strip().replace('_', '-').lower(), ""


def setup_requirements(req_file=None):
    target = _get_req_path(req_file)
    if not target:
        print(f"❌ Error: No requirements file found.")
        return
    
    print(f"--- 🛠️ Running Environment Setup via: {target} ---")
    cmd = [sys.executable, "-m", "pip", "install", "-r", target, "--no-warn-script-location"]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    noise = ["Requirement already satisfied", "Defaulting to user", "Looking in indexes"]
    changes_made = False
    for line in process.stdout:
        if not any(msg in line for msg in noise):
            print(f"  > {line.strip()}")
            changes_made = True

    if not changes_made: print("✅ No new installations required.")

    restart_required = False
    with open(target, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(('#', '-')): continue
            name, full_spec = _parse_line(line)
            try:
                curr_v = metadata.version(name).split('+')[0]
                if full_spec and Version(curr_v) not in SpecifierSet(full_spec):
                    print(f"🚨 Mismatch: {name} is {curr_v} (Expected {full_spec})")
                    restart_required = True
            except metadata.PackageNotFoundError:
                print(f"❌ Missing: {name}")
                restart_required = True

    if restart_required or changes_made:
        print("\n🛑 ACTION REQUIRED: Restart your Kernel to apply changes.")
    else:
        print("\n✅ Environment is synchronized.")

def check_requirements(file_path=None):
    target = _get_req_path(file_path)
    if not target: return None

    print(f"--- 📄 System Audit: {target} ---")
    with open(target, 'r') as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith(('#', '--'))]

    missing, mismatch, healthy = [], [], []
    for line in lines:
        name, full_spec = _parse_line(line)
        try:
            curr_v_str = metadata.version(name)
            clean_v = curr_v_str.split('+')[0] 
            if full_spec:
                if Version(clean_v) in SpecifierSet(full_spec): healthy.append(name)
                else: mismatch.append(f"{name} (Found {curr_v_str}, Need {full_spec})")
            else: healthy.append(name)
        except metadata.PackageNotFoundError: missing.append(name)

    print(f"✅ Optimal: {len(healthy)} packages")
    if missing: print(f"❌ Missing: {missing}")
    if mismatch: print(f"⚠️ Mismatches: {mismatch}")
    
    return "optimal" if not missing and not mismatch else "out_of_sync"

def manage_env(req_file=None):
    """
    Handles pip requirements ONLY. No hardware/framework checks.
    """
    target_file = _get_req_path(req_file)
    if not target_file: return False

    print(f"🔍 Initializing Environment Audit using: {target_file}")
    status = check_requirements(target_file)

    if status == "optimal":
        return True

    try:
        setup_requirements(target_file)
        return check_requirements(target_file) == "optimal"
    except Exception as e:
        print(f"❌ Critical error during setup: {e}")
        return False

def prepare_for_deployment():
    """
    Ensures env_setup.py and requirements.txt exist in the local project 
    folder for portability. Provides clear console feedback.
    """
    # This must point to the folder WHERE THIS FILE LIVES on your Dell    
    global_source = r"C:\Users\ashish\Documents\Ashish\anaconda\ai_lab_312"
    files_to_sync = ['env_setup.py', 'requirements.txt']
    current_dir = os.getcwd()
    
    # Identify where the CURRENTLY LOADED env_setup is coming from
    # This helps you see if Python is using your .pth link or the local file
    loaded_from = sys.modules[__name__].__file__
    is_local = current_dir in loaded_from

    print(f"📍 Active Script: {'LOCAL' if is_local else 'GLOBAL'} ({loaded_from})")

    # If we are already running the LOCAL version, we don't need to copy from Global
    # unless we want to check for updates.
    for f in files_to_sync:
        target = os.path.join(current_dir, f)
        source = os.path.join(global_source, f)

        # 1. First Time Copy (Global -> Local)
        if not os.path.exists(target):
            if os.path.exists(source):
                shutil.copy2(source, target)
                print(f"📦 Deployment: Copied {f} to local project folder.")
            else:
                print(f"⚠️ Warning: Master {f} not found at {global_source}")
            continue
        else:
            # IF local file exists, we do NOTHING. 
            # This allows local and global to diverge safely.
            print(f"✅ Local {f} detected.")


# --- COLAB DRIVE SYNC UTILITY ---
def sync_to_colab_drive(file_list, drive_folder='IITD_AIML', source_path='/content/'):
    """
    Generic Colab to Drive sync. 
    Args:
        file_list (list): Filenames to move (e.g., ['model.h5']).
        drive_folder (str): Destination folder name in 'My Drive'.
        source_path (str): Where files are currently located in Colab.
    """

    # 1. Exit early if not in Colab
    if 'google.colab' not in sys.modules:
        print("⚠️  Not running in Colab — skipping Drive sync.")
        return None

    from google.colab import drive

    print(f"--- ☁️ Colab Drive Sync: {drive_folder} ---")

    # 2. Mount Drive if not already mounted
    mount_point = '/content/drive'
    if not os.path.exists(os.path.join(mount_point, 'MyDrive')):
        drive.mount(mount_point)

    # 3. Define the local path to your Google Drive folder
    # This is the path Python uses to "see" your Drive
    drive_local_path = os.path.join(mount_point, 'MyDrive', drive_folder)

    # 4. Create destination folder if it doesn't exist
    os.makedirs(drive_local_path, exist_ok=True)
    print(f"📁 Destination: {drive_local_path}")

    # 5. Copy files
    for item in file_list:
        src = item if os.path.isabs(item) else os.path.join(source_path, item)
        dst = os.path.join(drive_local_path, os.path.basename(item))

        if os.path.isdir(src):
            try:
                shutil.copytree(src, dst, dirs_exist_ok=True)
                print(f"✅ Dir  copied : {item} → Drive")
            except Exception as e:
                print(f"❌ Error copying dir {item}: {e}")
        elif os.path.isfile(src):
            try:
                shutil.copy2(src, dst)
                print(f"✅ File copied : {item} → Drive")
            except Exception as e:
                print(f"❌ Error copying file {item}: {e}")
        else:
            print(f"⚠️  Not found  : {src}")

    print(f"📍 Local Drive Path: {drive_local_path}")
    print(f"🔗 Web Access: https://drive.google.com/drive/u/0/my-drive")
    
    return drive_local_path

# --- HARDWARE UTILITIES ---

def get_cpu_info():
    """Returns (physical_cores, logical_cores)."""
    l_cores = os.cpu_count() or 1
    p_cores = l_cores // 2 if l_cores > 1 else 1
    return p_cores, l_cores


# --- TENSORFLOW SPECIFIC UTILITIES ---

def apply_tf_threading(intra_threads=None, inter_threads=2):
    """Sets threading for TensorFlow. Must be called before TF initializes."""
    import tensorflow as tf
    if intra_threads is None:
        intra_threads, _ = get_cpu_info()
    try:
        tf.config.threading.set_intra_op_parallelism_threads(intra_threads)
        tf.config.threading.set_inter_op_parallelism_threads(inter_threads)
        return True
    except RuntimeError:
        return False

def setup_tensorflow(use_threading=True):
    """
    Initializes TensorFlow: Detects devices and applies threading.
    """
    import tensorflow as tf
    print(f"--- 💠 TensorFlow Setup ({platform.system()}) ---")

    # 1. Re-entrancy: Clear Keras global state
    tf.keras.backend.clear_session()
    print("🧹 Keras Global State: Cleared/Reset")

    # 2. Threading: Logic is already safe via try/except in apply_tf_threading
    if use_threading:
        p, _ = get_cpu_info()
        success = apply_tf_threading(p)
        status = f"Set to {p} cores" if success else "Already initialized/Late call"
        print(f"⚙️  Threading: {status}")

    # 2. Device Detection
    gpus = tf.config.list_physical_devices('GPU')
    cpus = tf.config.list_physical_devices('CPU')
    
    if gpus:
        for gpu in gpus:
            print(f"🚀 Device detected: {gpu}")
            # Try to print VRAM usage if memory growth is enabled
            try:
                # TensorFlow handles shared memory differently
                # but we can list the logical device if initialized.
                details = tf.config.experimental.get_virtual_device_configuration(gpu)
                if details:
                    print(f"📊 Configured Virtual Memory: {details}")
            except:
                pass
    else:
        print(f"🖥️  Device detected: {cpus[0] if cpus else 'None'}")

# --- PYTORCH SPECIFIC UTILITIES ---

def apply_torch_threading(intra_threads=None):
    """Sets threading for PyTorch. Safe for all devices."""
    import torch
    if intra_threads is None:
        intra_threads, _ = get_cpu_info()
    try:
        torch.set_num_threads(intra_threads)
        # Inter-op is generally best at 1 or 2 for research workloads
        torch.set_num_interop_threads(2) 
        return True
    except RuntimeError:
        return False

def torch_hard_reset():
    """Purges PyTorch memory pools and triggers garbage collection."""
    import torch
    gc.collect() 
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.empty_cache()
        torch.xpu.reset_peak_memory_stats()
        torch.xpu.synchronize() 
    print("🧹 PyTorch VRAM & RAM Purged.")

def get_pytorch_device(check_xpu=True):
    """Configures and returns the best available PyTorch device (CUDA, XPU, or CPU)."""
    import torch
  
    # 1. Priority: NVIDIA (Colab/Desktop)
    if torch.cuda.is_available():
        # Added name print for CUDA for consistency
        props = torch.cuda.get_device_properties(0)
        print(f"🚀 CUDA ACTIVE: {props.name}")
        print(f"📊 Dedicated VRAM: {props.total_memory / 1e9:.2f} GB")
        return torch.device("cuda")
    
    # 2. Priority: Intel (Dell XPU)
    if check_xpu and hasattr(torch, "xpu") and torch.xpu.is_available():
        # Apply Intel Integrated Graphics optimizations before driver lock-in
        os.environ["PYTORCH_XPU_ALLOC_CONF"] = "max_split_size_mb:512"
        os.environ["ZE_AFFINITY_MASK"] = "0" 
        torch.set_default_dtype(torch.float32)
        
        try:
            torch.xpu.empty_cache()
            torch.xpu.synchronize() 

            props = torch.xpu.get_device_properties(0)
            print(f"🚀 XPU ACTIVE: {props.name}")
            print(f"📊 Shared VRAM: {props.total_memory / 1e9:.2f} GB")
            return torch.device("xpu")
        except RuntimeError as e:
            if "DEVICE_LOST" in str(e):
                print("⚠️ XPU Backend Failure. Falling back to CPU.")
                return torch.device("cpu")
    
    print("🖥️ Mode: CPU")
    return torch.device("cpu")

def setup_pytorch(check_xpu=True, use_threading=True):
    """
    Initializes PyTorch with re-entrancy support.
    Detects device, prints status, and applies threading.
    Purges VRAM and prints current allocation status.
    Returns: torch.device
    """
    import torch
    print(f"--- 💠 PyTorch Setup ({platform.system()}) ---")

    # 1. Re-entrancy: Purge memory from previous run if torch was already active
    # (Checking torch.cuda.is_available is a safe way to check if backend is initialized)
    if torch.cuda.is_available() or (hasattr(torch, 'xpu') and torch.xpu.is_available()):
        torch_hard_reset()

    # 2. Threading
    if use_threading:
        p, _ = get_cpu_info()
        success = apply_torch_threading(p)
        status = f"Set to {p} cores" if success else f"Already initialized to {torch.get_num_threads()} cores"
        print(f"⚙️  Threading: {status}")

    # 3. Device Detection
    device = get_pytorch_device(check_xpu=check_xpu)

    # 4. Final Memory Usage Diagnostic
    try:
        if device.type == 'cuda':
            alloc = torch.cuda.memory_allocated(device) / 1e6
            print(f"📉 Current Usage: {alloc:.2f} MB")
        elif device.type == 'xpu':
            alloc = torch.xpu.memory_allocated(device) / 1e6
            print(f"📉 Current Usage: {alloc:.2f} MB")
    except Exception:
        pass # In case driver doesn't support usage stats

    return device
