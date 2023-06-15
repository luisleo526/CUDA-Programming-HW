if [ ! -f ./vecRedu ]
then
        make
fi

for t in 8 16 32 64 128 256 512
do
        for b in 32 64 128 256 512 1024
        do
                echo 0 81920007 $t $b | ./vecRedu
        done
done