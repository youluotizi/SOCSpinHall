## soc superfluid in optical lattice
首次运行本项目需注意：
1. 先打开 julia 更新一下自己的包。更新方法：先在 julia 运行
   `ENV["JULIA_PKG_SERVER"]="https://mirrors.cernet.edu.cn/julia"`  
   设置镜像；再按 `]`，进入包管理模式，输入`up` 回车更新
2. 切换环境。打开vscode，选择 `文件`-`打开文件夹`, 选择本项目的 "SOCSpinHall" 文件夹，此时左下角的 `julia env:` 后面应是 `SpinHall`, 若不是则点击然后选择为 `SpinHall`。
3. 实例化环境。删除原有的 "Manifest.toml" 文件；按 `alt`+`j`, `alt`+`o` 在 vscode 中打开 julia，然后按 `]`，输入 `instantiate` 回车，这会生成一个新的 "Manifest.toml" 文件。若提示缺的包就 `add missingPackage`。
4. 可以正常运行了。打开 "main.jl" 文件逐段执行
