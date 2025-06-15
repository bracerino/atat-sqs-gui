# Interactive interface for creating ATAT input files and analyzing outputs from crystal structure file
 Online application for generating and analyzing (SQS) (Special Quasi-Random Structures) using the ATAT mcsqs (Alloy Theoretic Automated Toolkit).

 ## Workflow
 - Upload crystal structures or retrieved them from implemented search interface in MP, AFLOW, and COD databases.
 - Generate ATAT Input Files: Create rndstr.in and sqscell.out files with global or sublattice-specific composition control. The concentrations are automatically recalculated for the achiavable concentrations given the total number of atoms in supercell.
 - Generate complete bash script that will create the rndstr.in and sqscell.out and run the ATAT mcsqs with improved monitoring and creating the mcsqs_progress.csv
 - Convert ATAT Output: Transform bestsqs.out files to VASP, CIF, LAMMPS, and XYZ formats with 3D visualization
 - Monitor Optimization: Analyze ATAT convergence from log files (mcsqs.log or at once for parallel run with mcsqs1.log, mcsqs2.log, ...) and CSV time progress file (mcsqs_progress.csv) data with interactive plots and main information. For parallel run, it will automatically identify the best performing optimization
 - Calculate PRDF: Compute Partial Radial Distribution Functions for SQS
 - Create Vacancy Structures: Generate ordered defects by selectively removing specific element from the SQS 

 
# **How to compile and run the ATAT-SQS-GUI locally:** 

### **Prerequisities**: 
- Python 3.x (Tested 3.12)
- Console (For Windows, I recommend to use WSL2 (Windows Subsystem for Linux))
- Git (optional for downloading the code)
  


### **Compile the app**  
Open your terminal console and write the following commands (the bold text):  
(Optional) Install Git:  
      **sudo apt update**  
      **sudo apt install git**    
      
1) Download the XRDlicious code from GitHub (or download it manually without Git on the following link by clicking on 'Code' and 'Download ZIP', then extract the ZIP. With Git, it is automatically extracted):  
      **git clone https://github.com/bracerino/xrdlicious.git**

2) Navigate to the downloaded project folder:  
      **cd xrdlicious/**

3) Create a Python virtual environment to prevent possible conflicts between packages:  
      **python3 -m venv xrdlicious_env**

4) Activate the Python virtual environment (before activating, make sure you are inside the xrdlicious folder):  
      **source xrdlicious_env/bin/activate**
   
5) Install all the necessary Python packages:  
      **pip install -r requirements.txt**

6) Run the XRDlicious app (always before running it, make sure to activate its Python virtual environment (Step 4):  
      **streamlit run app.py**
