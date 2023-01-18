#! /bin/bash

wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
unzip tiny-imagenet-200.zip 

datadir="$(pwd)/tiny-imagenet-200"
annotation_file="val_annotations.txt"

cd $datadir/train

for dir in $(ls)
do
   cd $dir
   rm *.txt
   mv images/* .
   rm -r images
   cd ..
done

cd $datadir/val
img_count=$(cat $annotation_file | wc -l)

for i in $(seq 1 $img_count)
do
    line=$(sed -n ${i}p $annotation_file)
    image=$(echo $line | cut -f1 -d" " )
    dir=$(echo $line | cut -f2 -d" ")
    mkdir -p $dir
    mv images/$image $dir
done

rm -r images
cd ..
rm *.txt