## soc superfluid in optical lattice
首次运行本项目需注意：
1. 先打开 julia 更新一下自己的包。更新方法：先在 julia 输入
   `ENV["JULIA_PKG_SERVER"]="https://mirrors.cernet.edu.cn/julia"`  
   设置镜像，再按 `]`，进入包管理模式，输入`up` 回车
2. vscode 打开 SpinHall 文件夹，删除 "Manifest.toml" 文件
2. 然后按 `alt`+`j`, `alt`+`o`，在 vscode 中打开 julia，然后按 `]`，输入 `instantiate` 回车，显示缺的包就 `add missingPackage`。
3. 打开 "main.jl" 文件逐段执行
