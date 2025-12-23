# LaTeXmk配置文件 - 使用XeLaTeX编译中文文档
# 原因：ctex使用fontset=none时需要XeLaTeX支持

$pdf_mode = 5;  # 5表示使用xelatex，1表示pdflatex
$xelatex = 'xelatex -synctex=1 -interaction=nonstopmode -file-line-error %O %S';
$bibtex_use = 2;  # 运行bibtex
$out_dir = 'build';  # 输出目录
$clean_ext = 'synctex.gz synctex.gz(busy) run.xml tex.bak bbl bcf fdb_latexmk run tdo %R-blx.bib';
