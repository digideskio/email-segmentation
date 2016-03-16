To segment an email into following three parts based on statistical model coupled with
some rudimentary rules: 


1)Reply 

2)Signature and 

3)Other


Related Literature: http://www.cs.cmu.edu/~wcohen/postscript/email-2004.pdf

Spacy documentation: http://spacy.io/#example-use


Clone the repository, install all the dependencies and download English corpora with following command:
	
	$ python -m spacy.en.download all

And run with:

	$ python scripts/segmentation.py