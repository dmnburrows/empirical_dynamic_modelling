#!/bin/bash

#Define input variables
export HDF5_DISABLE_VERSION_CHECK=1


y=0
#Define list
datapath="/snl/scratch25/dburrows/CCM/smoothed/sigma_2/"
cd $datapath
array=($(ls *trace*pre-CCM*.h5* ))

#Loop through and run kEDM
for i in "${array[@]}"
do
  echo "Running $i"
  # print a counter of how many files have been processed out of total
  ((y=y+1))

  echo "Processing file $y of ${#array[@]}"

  filename="$i"
  savename="${filename/_trace_pre-CCM.h5/_CCMxmap.h5}"

  echo $filename
  echo $savename

  edm-xmap -d, --dataset "data" --rho --rho-diff $filename $savename
  
done

echo "Finished!"



y=0
#Define list
datapath="/snl/scratch25/dburrows/CCM/smoothed/sigma_4/"
cd $datapath
array=($(ls *trace*pre-CCM*.h5* ))

#Loop through and run kEDM
for i in "${array[@]}"
do
  echo "Running $i"
  # print a counter of how many files have been processed out of total
  ((y=y+1))

  echo "Processing file $y of ${#array[@]}"

  filename="$i"
  savename="${filename/_trace_pre-CCM.h5/_CCMxmap.h5}"

  echo $filename
  echo $savename

  edm-xmap -d, --dataset "data" --rho --rho-diff $filename $savename
  
done

echo "Finished!"





