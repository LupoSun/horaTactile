# Source before Isaac Gym on WSL2:
#   source scripts/isaac_wsl_env.sh
# 1. PhysX needs libcuda from the WSL driver path.
# 2. Vulkan viewer needs the Mesa dozen (dzn) ICD to see the GPU.
if [ -d /usr/lib/wsl/lib ]; then
  export LD_LIBRARY_PATH="/usr/lib/wsl/lib:${LD_LIBRARY_PATH}"
fi
if [ -f /usr/share/vulkan/icd.d/dzn_icd.json ]; then
  export VK_ICD_FILENAMES="/usr/share/vulkan/icd.d/dzn_icd.json"
fi
