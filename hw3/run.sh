if [ ! -f ./poisson3d ]
then
	make
fi

for size in 8 16 32 64
do
	echo $size 64 16 | ./poisson3d
done