#! /bin/bash

for fin in x-*hpp 
do
  func=$(grep RES_T $fin | sed  -e "s/(/ /" | awk '{print $2}')
  printf "%-48s %s\n" $func $fin
  #printf "%-48s\n" $func
done

