if [ ! -f ./monte_carlo ]
then
	make
fi

for t in 4 8 16 32 64 128 256 512
do
	echo $t  | ./monte_carlo

done