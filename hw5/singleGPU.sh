if [ ! -f ./laplace2d_NGPU ]
then
    make
fi

for t in 4 8 16 32 64 128 256
do
    echo 1 1 $t 1 | ./laplace2d_NGPU
done

# for t in 4 8 16 32 64 128 256
# do
#     echo 1 1 $t 1 | ./laplace2d_NGPU
# done