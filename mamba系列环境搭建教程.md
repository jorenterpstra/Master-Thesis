## 项目环境搭建指南
*Project Environment Setup Guide*

### 1. 环境要求
*1. Environment Requirements*

建议在 Windows 系统上使用 WSL2 进行开发。
*It is recommended to use WSL2 on Windows for development.*

### 2. 创建和激活 Conda 虚拟环境
*2. Create and Activate Conda Virtual Environment*

在终端中依次输入以下命令：
*Enter the following commands in the terminal:*

```bash
conda create -n mamba python=3.10
conda activate mamba
```

### 3. 安装 CUDA 工具包和 PyTorch
*3. Install CUDA Toolkit and PyTorch*

继续在终端中输入以下命令：
*Continue to enter the following commands in the terminal:*

```bash
conda install cudatoolkit==11.8 -c nvidia
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118
```

### 4. 安装 CUDA 编译器和其他依赖
*4. Install CUDA Compiler and Other Dependencies*

输入以下命令来安装 CUDA 编译器和其他依赖：
*Enter the following commands to install the CUDA compiler and other dependencies:*

```bash
conda install -c "nvidia/label/cuda-11.8.0" cuda-nvcc
conda install packaging
```

### 5. 克隆项目仓库
*5. Clone Project Repository*

在终端中运行以下命令来克隆项目：
*Run the following command in the terminal to clone the project:*

```bash
git clone https://github.com/hustvl/Vim.git
```

### 6. 进入项目目录
*6. Enter Project Directory*

输入以下命令切换到项目目录：
*Enter the following command to switch to the project directory:*

```bash
cd Vim
```

### 7. 安装项目依赖
*7. Install Project Dependencies*

使用 `pip` 安装项目依赖：
*Use `pip` to install project dependencies:*

```bash
pip install -r vim/vim_requirements.txt
```

> **提示：** 如果安装速度慢，可以使用国内镜像源，如清华大学镜像源：
> *Tip: If the installation is slow, you can use a domestic mirror source, such as Tsinghua University mirror source:*
> ```bash
> pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
> ```

### 8. 解决可能的 CUDA 环境问题
*8. Resolve Possible CUDA Environment Issues*

在安装 `causal_conv1d` 时，可能会遇到如下报错：
*When installing `causal_conv1d`, you may encounter the following error:*

```
No such file or directory: ‘:/usr/local/cuda-11.8/bin/nvcc‘
```
解决方式是:更改bashrc文件, 设置 `CUDA_HOME` 环境变量,在文件最下面添加下面代码
*The solution is to change the bashrc file, set the `CUDA_HOME` environment variable, and add the following code at the bottom of the file:*
export PATH=/root/anaconda3/envs/mamba/bin:$PATH
export CUDA_HOME=/root/anaconda3/envs/mamba
export LD_LIBRARY_PATH=/root/anaconda3/envs/mamba/lib:$LD_LIBRARY_PATH

下面这个方法不行
*The following method does not work:*
```bash
export CUDA_HOME=/usr/local/cuda
```

然后安装 `causal_conv1d`：
*Then install `causal_conv1d`:*

```bash
pip install causal_conv1d>=1.1.0
```
直接安装 `pip install causal_conv1d==1.1.3`
*Directly install `pip install causal_conv1d==1.1.3`*

`causal_conv1d-1.1.3+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl` 文件
*`causal_conv1d-1.1.3+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl` file*

### 9. 安装 `mamba-ssm`
*9. Install `mamba-ssm`*

有两种方法可以安装 `mamba-ssm`：
*There are two ways to install `mamba-ssm`:*

#### 方法一：使用 `pip` 安装
*Method 1: Install using `pip`*
```bash
pip install mamba-ssm
```

#### 方法二：从源码安装
*Method 2: Install from Source*

先克隆 `mamba` 项目，然后安装：
*First clone the `mamba` project, then install:*

```bash
git clone https://github.com/state-spaces/mamba.git
cd mamba
pip install .
```

之后首先找到你所创建的环境里的mamba ssm文件夹，路径一般在这里："/envs/mamba/lib/python3.10/site-packages/mamba_ssm/"，将其替换为你项目代码里的mamba_ssm文件夹：
*Then first find the mamba ssm folder in the environment you created, the path is generally here: "/envs/mamba/lib/python3.10/site-packages/mamba_ssm/", replace it with the mamba_ssm folder in your project code:*

rm -rf "/envs/mamba/lib/python3.10/site-packages/mamba_ssm/" (删除原环境里的mamba_ssm)
*rm -rf "/envs/mamba/lib/python3.10/site-packages/mamba_ssm/" (delete the original mamba_ssm in the environment)*

cp -r "Vim-main/mamba-1p1p1/mamba_ssm" "/envs/mamba/lib/python3.10/site-packages/" (将Vim项目code里的mamba_ssm文件夹替换进去)
*cp -r "Vim-main/mamba-1p1p1/mamba_ssm" "/envs/mamba/lib/python3.10/site-packages/" (replace the mamba_ssm folder in the Vim project code)*

完成以上步骤后，你的环境应该已经搭建完毕，能够正常运行项目了。
*After completing the above steps, your environment should be set up and the project should run normally.*