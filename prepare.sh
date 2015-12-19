#!/bin/sh
for file in ./*wav
do
    sox $file ${file/.wav/_mono.wav} remix 1
    rm $file
    sox ${file/.wav/_mono.wav} -r 22k $file
    rm ${file/.wav/_mono.wav}
done
