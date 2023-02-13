
# 1. 前言

- 所有命令都以\\开头，后面带{}表示该命令的参数
```latex
# 指定文档类型，显示中文需要ctexart文档类型
\documentclass[UTF8]{article}

\title{文档的标题}
\author{作者名}

# 修改日期
\date{\today}

# 添加图片
\usepackage{graphicx}


# 前言
\begin{document}
# 当前位置生成文档标题
\maketitle
# begin和end中间的叫做正文

\end{document}
```

# 2. 格式化命令﻿

```LaTeX
# 加粗
\textbf{}

# 斜体
\texttit{}

# 下划线
\underline{}

# 新章节
\section{}

# 二级章节
\subsection{}

# 三级章节
\subsubsection{}
```

# 3. 图片命令

```LaTeX
# 在当前位置添加1张图片
\includegraphics[width=0.5\textwidth]{}

# 给图片添加标题
\begin{figure}
\centering # 居中显示
\includegraphics[width=0.5\textwidth] {head}
\caption{图片标题}
\end{figure}
```

# 4. 列表命令

```LaTeX
# 无序列表
\begin{itemize}
\item列表项1
\end{itemize}

# 有序列表
\begin{enumerate}
\item列表项1
\end{enumerate}
```

# 5. 公式

```LaTeX
# 行内公式
$公式内容$

# 单行公式
\begin{equation}
公式内容
\end{equation}

# 以上简写
\[
公式内容
\]

```

在线公式编辑器：
https://latex.codecogs.com/eqneditor/editor.php

# 6. 表格

```LaTeX
# 三列表格且居中对齐c，l左对齐，r右对齐
# 内容 & 隔开
\begin{tabular}{ c c c }
单元格1 & 单元格2 & 单元格3 \\
单元格4 & 单元格5 & 单元格6 \\
\end{tabular}

# | 竖直方向边框
# \hline 水平方向边框
# p{每列宽度}
\begin{tabular}{| p{2cm}| c| c| }
\hline
单元格1 &单元格2 &单元格3 \\
\hline\hline
单元格4 &单元格5 &单元格6 \\
\hline
单元格7 &单元格8 &单元格9 \\
\hline
\end{tabular}

# 添加标题
同图片

```

# 7. 其他资料

- 中文参考手册

https://github.com/CTeX-org/lshort-zh-cn

- latex模板
IEEE模板:
https://template-selector.ieee.org/secure/templateSelector/publicationType


# 参考文献

- [一个非常快速的 Latex 入门教程](https://www.bilibili.com/video/BV11h41127FD/?spm_id_from=333.337.search-card.all.click&vd_source=9ea40c1ac510f1e604555a3c8278ff94)
- [26 用LaTeX写期刊论文的详细教程](https://www.bilibili.com/video/BV1JR4y1H7Es/?spm_id_from=333.337.search-card.all.click&vd_source=9ea40c1ac510f1e604555a3c8278ff94)
