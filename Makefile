deploy: 
	jupyter-book build .
	netlify deploy --dir=_build/html --prod
build: 
	jupyter-book build .
	open -a "Google Chrome" _build/html/index.html

