if [ ! -f ./vecDot_NGPU ]
then
	make
fi

for t in 4 8 16 32 64 128 256
do
	for b in 64 128 256 512 1024 2048
	do
		echo $1 4096000 $t $b | ./vecDot_NGPU
	done
done