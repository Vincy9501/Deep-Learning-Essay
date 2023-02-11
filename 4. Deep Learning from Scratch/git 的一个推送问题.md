![[Pasted image 20230211114707.png]]
提示文件大小超过100MB

1. 安装lfs
```git
git lfs install
```

2. migrate import
`git lfs migrate` <模式> [选项][--][分支...]
该模式将git文件转换为git lfs
```git
 git lfs migrate import --include="*.csv"
```
![[Pasted image 20230211115129.png]]

3. `git push`
![[Pasted image 20230211115201.png]]

另一种方式：

1. `git lfs track "mnist.pkl"`
2. `git add mnist.pkl`
3. ` git commit -m "2023-02-11 mnist.pkl"`

# 参考文献

- [[Git] 处理 github 不允许上传超过 100MB 文件的问题](http://www.liuxiao.org/2017/02/git-%E5%A4%84%E7%90%86-github-%E4%B8%8D%E5%85%81%E8%AE%B8%E4%B8%8A%E4%BC%A0%E8%B6%85%E8%BF%87-100mb-%E6%96%87%E4%BB%B6%E7%9A%84%E9%97%AE%E9%A2%98/)
-  [【git】git利用git-lfs提交大文件](https://www.cnblogs.com/vickylinj/p/16207086.html)
- [Github超过100M的大文件上传](https://www.jianshu.com/p/7d8003ba2324)