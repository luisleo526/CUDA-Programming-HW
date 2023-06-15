if [ ! -f ./hist ]
then
	make
fi

for t in 4 8 16 32 64 128 256
do
	echo 0 4096000 $t $b | ./hist

done

for t in 4 8 16 32 64 128 256
do
	echo 1 4096000 $t $b | ./hist

done