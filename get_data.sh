
echo "Getting the freebase kb"
wget -O kb/freebase.spades.txt http://iesl.cs.umass.edu/downloads/spades/freebase.spades.txt
if [ $? -ne 0 ]; then
	echo "Failed to get the kb"
	echo "exiting..."
	exit 1
fi
echo "Getting the text kb"

wget -O text_kb/text_kb.spades.txt http://iesl.cs.umass.edu/downloads/spades/text_kb.spades.txt
if [ $? -ne 0 ]; then
	echo "Failed to get the kb"
	echo "exiting..."
	exit 1
fi
echo "Done..."
