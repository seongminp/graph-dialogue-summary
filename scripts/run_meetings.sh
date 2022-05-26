echo "AMI" &&
python benchmarks/ami.py -d data/ami &&

echo "AMI (pre-Segmented)" &&
python benchmarks/ami_segmented.py -d data/ami -s data/ami_segmented &&

echo "AMI (POV conversion)" &&
python benchmarks/ami.py -d data/ami -c &&

echo "AMI (pre-Segmented, POV conversion)" &&
python benchmarks/ami_segmented.py -d data/ami -s data/ami_segmented -c &&

echo "ICSI" &&
python benchmarks/icsi.py -d data/icsi &&

echo "ICSI (pre-Segmented)" &&
python benchmarks/icsi_segmented.py -d data/icsi -s data/icsi_segmented &&

echo "ICSI (POV conversion)" &&
python benchmarks/icsi.py -d data/icsi -c &&

echo "ICSI (pre-Segmented, POV conversion)" &&
python benchmarks/icsi_segmented.py -d data/icsi -s data/icsi_segmented -c
