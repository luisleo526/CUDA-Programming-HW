if [ ! -f ./vecAdd ]
then
	make
fi

for size in 4 8 10 16 20 32
do
	echo 0 6400 $size | ./vecAdd
done