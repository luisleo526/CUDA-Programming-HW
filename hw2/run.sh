if [ ! -f ./vecRedu ]
then
	make
fi

for size in 8 16 32 64 128 256 512
do
	echo 0 8192007 $size | ./vecRedu
done