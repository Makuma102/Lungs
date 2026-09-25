#!/bin/bash
# Resumable: TotalSegmentator fast lobes+trachea for every MSD training case.
ROI="lung_upper_lobe_left lung_lower_lobe_left lung_upper_lobe_right lung_middle_lobe_right lung_lower_lobe_right trachea"
for p in data/Task06_Lung/imagesTr/lung_*.nii.gz; do
  c=$(basename $p .nii.gz); o=data/anat/$c/total
  [ "$(ls $o 2>/dev/null | wc -l)" -ge 6 ] && continue
  rm -rf $o
  timeout 900 TotalSegmentator -i $p -o $o --fast --roi_subset $ROI -d cpu --nr_thr_resamp 1 --nr_thr_saving 1 > /dev/null 2>&1
  echo "$c $(ls $o 2>/dev/null | wc -l)"
done
echo DONE
