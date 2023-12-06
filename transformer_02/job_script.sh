#!/bin/bash
### Job Name
#PBS -N python_script
### Required runtime
#PBS -l walltime=00:10:00
### Queue for submission
#PBS -q student

### Merge output and error files
#PBS -j oe

### Request 16 GB of memory and 1 CPU core on 1 compute node
#PBS -l select=1:mem=16G:ncpus=1

### Start the job in the directory it was submitted from
cd $PBS_O_WORKDIR

# Activate the Python virtual environment:
# Note that once activated, the Python version used when creating
# the virtual environment is readily available, and the respective module
# need not be loaded by means of "module load python/X.X.X".
source env/bin/activate

### Run the application
python transform2trans.py