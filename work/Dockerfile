FROM texlive/texlive:latest

# main.tex (and the compiled output) must be volume-mounted at /work
WORKDIR /work

# Continuously recompile main.tex whenever it changes
CMD ["latexmk", "-pdf", "-pvc", "-interaction=nonstopmode", "-halt-on-error", "main.tex"]
