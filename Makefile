all: *.tex *.bib
	latexmk -pdf -bibtex *.tex

clean:
	latexmk -C && rm -f *.bbl *.bak

show:
	latexmk -pdf -bibtex *.tex && xdg-open *.pdf
