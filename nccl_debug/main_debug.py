#!/usr/bin/env python3
import os
import subprocess
import socket
import torch
import torch.distributed as dist
import argparse
import sys
import platform
from datetime import datetime

def run_command(cmd):
    """Run a shell command and return its output"""
    try:
        result = subprocess.run(cmd, shell=True, check=True, 
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                universal_newlines=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        return f"Command failed with exit code {e.returncode}: {e.stderr}"

def print_section(title):
    """Print a section title"""
    print("\n" + "="*80)
    print(f" {title}")
    print("="*80)

def check_system_info():
    print_section("System Information")
    
    print(f"Hostname: {socket.gethostname()}")
    print(f"Platform: {platform.platform()}")
    print(f"Python version: {sys.version}")
    print(f"Date and time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check if running in container
    cgroup = run_command("cat /proc/self/cgroup")
    if "docker" in cgroup or "container" in cgroup:
        print("Container detected: Yes")
    else:
        print("Container detected: No")

def check_gpu_info():
    print_section("GPU Information")
    
    # Check NVIDIA drivers
    nvidia_smi = run_command("nvidia-smi")
    print(f"NVIDIA-SMI output:\n{nvidia_smi}\n")
    
    # PyTorch CUDA information
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

def check_network_interfaces():
    print_section("Network Interface Information")
    
    # List network interfaces
    if_output = run_command("ip addr show")
    print(f"Network interfaces:\n{if_output}\n")
    
    # Check if IP addresses are assigned
    ip_output = run_command("hostname -I")
    print(f"IP Addresses: {ip_output}\n")

    # Check routing
    route_output = run_command("ip route")
    print(f"Routing table:\n{route_output}\n")

def check_nccl_info():
    print_section("NCCL Information")
    
    # Check NCCL version from PyTorch
    if hasattr(torch.cuda, 'nccl'):
        print(f"NCCL Version from PyTorch: {torch.cuda.nccl.version() if hasattr(torch.cuda.nccl, 'version') else 'Not available'}")
    else:
        print("NCCL info not available from PyTorch")
    
    # Check NCCL environment variables
    nccl_env_vars = {
        "NCCL_DEBUG": os.environ.get("NCCL_DEBUG", "Not set"),
        "NCCL_SOCKET_IFNAME": os.environ.get("NCCL_SOCKET_IFNAME", "Not set"),
        "NCCL_IB_DISABLE": os.environ.get("NCCL_IB_DISABLE", "Not set"),
        "NCCL_P2P_DISABLE": os.environ.get("NCCL_P2P_DISABLE", "Not set"),
        "NCCL_SHM_DISABLE": os.environ.get("NCCL_SHM_DISABLE", "Not set"),
    }
    
    print("NCCL Environment Variables:")
    for var, value in nccl_env_vars.items():
        print(f"  {var}: {value}")

def test_simple_nccl():
    print_section("Simple NCCL Test")
    
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        print("Skipping NCCL test: No CUDA devices available")
        return
    
    try:
        # Set environment variables for debugging
        os.environ["NCCL_DEBUG"] = "INFO"
        
        # Initialize process group
        print("Attempting to initialize process group with NCCL backend...")
        dist.init_process_group(
            backend="nccl",
            init_method="tcp://127.0.0.1:12345",
            rank=0,
            world_size=1
        )
        
        # Create tensor on GPU
        tensor = torch.randn(10, device="cuda")
        
        # Perform a simple all-reduce operation
        print("Attempting all-reduce operation...")
        dist.all_reduce(tensor)
        
        print("NCCL test completed successfully!")
        
        # Clean up
        dist.destroy_process_group()
        
    except Exception as e:
        print(f"NCCL test failed with error: {e}")

def recommend_fixes():
    print_section("Recommendations")
    
    print("Based on the 'no socket interface found' error, try the following fixes:")
    print("1. Set NCCL_DEBUG=INFO to get more detailed error messages:")
    print("   export NCCL_DEBUG=INFO")
    print("\n2. Explicitly specify which network interface to use:")
    print("   export NCCL_SOCKET_IFNAME=eth0,en0,ib0")
    print("\n3. If running in a container, ensure the container has proper network visibility")
    print("\n4. For troubleshooting, try disabling InfiniBand:")
    print("   export NCCL_IB_DISABLE=1")
    print("\n5. Ensure all nodes can communicate with each other:")
    print("   Try 'ping' between the nodes")
    print("\n6. Check if firewall rules are blocking NCCL communication")
    
    print("\nFor more information, refer to the NCCL documentation:")
    print("https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting.html")

def parse_args():
    parser = argparse.ArgumentParser(description="Debug NCCL Issues")
    parser.add_argument("--skip-nccl-test", action="store_true", 
                        help="Skip the NCCL test (useful if it crashes)")
    return parser.parse_args()

def main():
    args = parse_args()
    
    print("\nNCCL Debugging Tool")
    print("Running diagnostics...\n")
    
    check_system_info()
    check_gpu_info()
    check_network_interfaces()
    check_nccl_info()
    
    if not args.skip_nccl_test:
        test_simple_nccl()
    
    recommend_fixes()
    
    print("\nDiagnostics completed. Check the output above for insights into your NCCL issues.")

if __name__ == "__main__":
    main()
