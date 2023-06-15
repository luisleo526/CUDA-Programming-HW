if [ ! -f ./hist ]
then
	make
fi

for t in 4 8 16 32 64 128 256
do
	echo 0 $t 500 | ./hist

done

for t in 4 8 16 32 64 128 256
do
	echo 1 $t | ./hist

done